# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import torchaudio
import argparse
from pathlib import Path


from text.g2p_module import G2PModule
from utils.tokenizer import AudioTokenizer, tokenize_audio
from models.tts.valle.valle import VALLE
from models.tts.base.tts_inferece import TTSInference
from models.tts.valle.valle_dataset import VALLETestDataset, VALLETestCollator
from processors.phone_extractor import phoneExtractor
from text.text_token_collation import phoneIDCollation
from models.tts.valle.speaker_encoder import ECAPASpeakerEncoder
from models.vocoders.diffusion.diffwave.token_diffwave import TokenConditionedDiffWave


class VALLEInference(TTSInference):
    def __init__(self, args=None, cfg=None):
        TTSInference.__init__(self, args, cfg)

        self.g2p_module = G2PModule(backend=self.cfg.preprocess.phone_extractor)
        text_token_path = os.path.join(
            cfg.preprocess.processed_dir, cfg.dataset[0], cfg.preprocess.symbols_dict
        )
        self.audio_tokenizer = AudioTokenizer()
        self.speaker_encoder = None
        self.token_diffwave = None
        if getattr(self.cfg.preprocess, "use_spk_embedding", False):
            source = getattr(
                self.cfg.preprocess,
                "speaker_encoder_source",
                "speechbrain/spkrec-ecapa-voxceleb",
            )
            savedir = getattr(
                self.cfg.preprocess,
                "speaker_encoder_savedir",
                "pretrained/ecapa_voxceleb",
            )
            self.speaker_encoder = ECAPASpeakerEncoder(
                source=source,
                savedir=savedir,
                device=str(self.device),
            )
        if hasattr(self.cfg.model, "token_diffwave") and self.args.vocoder_dir:
            self.token_diffwave = TokenConditionedDiffWave(
                self.cfg.model.token_diffwave
            ).to(self.device)
            checkpoint_path = Path(self.args.vocoder_dir)
            if checkpoint_path.is_dir():
                candidates = sorted(
                    list(checkpoint_path.glob("*.pt"))
                    + list(checkpoint_path.glob("*.pth")),
                    key=lambda path: path.stat().st_mtime,
                )
                if not candidates:
                    raise FileNotFoundError(
                        f"No Token-DiffWave checkpoint in {checkpoint_path}"
                    )
                checkpoint_path = candidates[-1]
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get(
                "model", checkpoint.get("state_dict", checkpoint)
            )
            self.token_diffwave.load_state_dict(state_dict)
            self.token_diffwave.eval()

    def _build_model(self):
        model = VALLE(self.cfg.model)
        return model

    def _build_test_dataset(self):
        return VALLETestDataset, VALLETestCollator

    def inference_one_clip(self, text, text_prompt, audio_file, save_name="pred"):
        reference_info = torchaudio.info(audio_file)
        reference_duration = reference_info.num_frames / reference_info.sample_rate
        if not 3.0 <= reference_duration <= 10.0:
            raise ValueError(
                f"Reference audio must be 3-10 seconds, got {reference_duration:.2f}s"
            )
        # get phone symbol file
        phone_symbol_file = None
        if self.cfg.preprocess.phone_extractor != "lexicon":
            phone_symbol_file = os.path.join(
                self.exp_dir, self.cfg.preprocess.symbols_dict
            )
            assert os.path.exists(phone_symbol_file)
        # convert text to phone sequence
        phone_extractor = phoneExtractor(self.cfg)
        # convert phone sequence to phone id sequence
        phon_id_collator = phoneIDCollation(
            self.cfg, symbols_dict_file=phone_symbol_file
        )

        text = f"{text_prompt} {text}".strip()
        phone_seq = phone_extractor.extract_phone(text)  # phone_seq: list
        phone_id_seq = phon_id_collator.get_phone_id_sequence(self.cfg, phone_seq)
        phone_id_seq_len = torch.IntTensor([len(phone_id_seq)]).to(self.device)

        # convert phone sequence to phone id sequence
        phone_id_seq = np.array([phone_id_seq])
        phone_id_seq = torch.from_numpy(phone_id_seq).to(self.device)

        # extract acoustic token
        encoded_frames = tokenize_audio(self.audio_tokenizer, audio_file)
        audio_prompt_token = encoded_frames[0][0].transpose(2, 1).to(self.device)
        speaker_embedding = None
        if self.speaker_encoder is not None:
            speaker_embedding = self.speaker_encoder.encode_file(audio_file).to(
                self.device
            )

        # copysyn
        if self.args.copysyn:
            samples = self.audio_tokenizer.decode(encoded_frames)
            audio_copysyn = samples[0].cpu().detach()

            out_path = os.path.join(
                self.args.output_dir, self.infer_type, f"{save_name}_copysyn.wav"
            )
            torchaudio.save(out_path, audio_copysyn, self.cfg.preprocess.sampling_rate)

        if self.args.continual:
            encoded_frames = self.model.continual(
                phone_id_seq,
                phone_id_seq_len,
                audio_prompt_token,
            )
        else:
            enroll_x_lens = None
            if text_prompt:
                # prompt_phone_seq = tokenize_text(self.g2p_module, text=f"{text_prompt}".strip())
                # _, enroll_x_lens = self.text_tokenizer.get_token_id_seq(prompt_phone_seq)

                text = f"{text_prompt}".strip()
                prompt_phone_seq = phone_extractor.extract_phone(
                    text
                )  # phone_seq: list
                prompt_phone_id_seq = phon_id_collator.get_phone_id_sequence(
                    self.cfg, prompt_phone_seq
                )
                prompt_phone_id_seq_len = torch.IntTensor(
                    [len(prompt_phone_id_seq)]
                ).to(self.device)

            encoded_frames = self.model.inference(
                phone_id_seq,
                phone_id_seq_len,
                audio_prompt_token,
                enroll_x_lens=prompt_phone_id_seq_len,
                top_k=self.args.top_k,
                temperature=self.args.temperature,
                speaker_embedding=speaker_embedding,
            )

        if self.token_diffwave is not None:
            if speaker_embedding is None:
                raise ValueError("Token-DiffWave inference requires speaker embedding")
            audio = self.token_diffwave.sample(
                encoded_frames,
                speaker_embedding,
            )[0].cpu()
        else:
            samples = self.audio_tokenizer.decode(
                [(encoded_frames.transpose(2, 1), None)]
            )
            audio = samples[0].squeeze(0).cpu().detach()

        return audio

    def inference_for_single_utterance(self):
        text = self.args.text
        text_prompt = self.args.text_prompt
        audio_file = self.args.audio_prompt

        if not self.args.continual:
            assert text != ""
        else:
            text = ""
        assert text_prompt != ""
        assert audio_file != ""

        audio = self.inference_one_clip(text, text_prompt, audio_file)

        return audio

    def inference_for_batches(self):
        test_list_file = self.args.test_list_file
        assert test_list_file is not None

        pred_res = []
        with open(test_list_file, "r") as fin:
            for idx, line in enumerate(fin.readlines()):
                fields = line.strip().split("|")
                if self.args.continual:
                    assert len(fields) == 2
                    text_prompt, audio_prompt_path = fields
                    text = ""
                else:
                    assert len(fields) == 3
                    text_prompt, audio_prompt_path, text = fields

                audio = self.inference_one_clip(
                    text, text_prompt, audio_prompt_path, str(idx)
                )
                pred_res.append(audio)

        return pred_res

        """
        TODO: batch inference 
        ###### Construct test_batch ######
        n_batch = len(self.test_dataloader)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(
            "Model eval time: {}, batch_size = {}, n_batch = {}".format(
                now, self.test_batch_size, n_batch
            )
        )       
        
        ###### Inference for each batch ######
        pred_res = []
        with torch.no_grad():
            for i, batch_data in enumerate(
                self.test_dataloader if n_batch == 1 else tqdm(self.test_dataloader)
            ):
                if self.args.continual:
                    encoded_frames = self.model.continual(
                        batch_data["phone_seq"],
                        batch_data["phone_len"],
                        batch_data["acoustic_token"],
                    )
                else:
                    encoded_frames = self.model.inference(
                        batch_data["phone_seq"],
                        batch_data["phone_len"],
                        batch_data["acoustic_token"],
                        enroll_x_lens=batch_data["pmt_phone_len"],
                        top_k=self.args.top_k,
                        temperature=self.args.temperature
                    )
                    
                samples = self.audio_tokenizer.decode(
                    [(encoded_frames.transpose(2, 1), None)]
                )
                
                            
                for idx in range(samples.size(0)):
                    audio = samples[idx].cpu()
                    pred_res.append(audio)
                    
        return pred_res
        """

    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--text_prompt",
            type=str,
            default="",
            help="Text prompt that should be aligned with --audio_prompt.",
        )

        parser.add_argument(
            "--audio_prompt",
            type=str,
            default="",
            help="Audio prompt that should be aligned with --text_prompt.",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=-100,
            help="Whether AR Decoder do top_k(if > 0) sampling.",
        )

        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="The temperature of AR Decoder top_k sampling.",
        )

        parser.add_argument(
            "--continual",
            action="store_true",
            help="Inference for continual task.",
        )

        parser.add_argument(
            "--copysyn",
            action="store_true",
            help="Copysyn: generate audio by decoder of the original audio tokenizer.",
        )
