import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path

try:
    from pystoi import stoi as compute_stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


class AudioAdversarialAttack:
    def __init__(self, model, processor, epsilon=0.1, lr=1e-3, device=None,
                 lambda_l2=0.0, lambda_stoi=0.0, stoi_threshold=0.50, stoi_check_interval=50,
                 use_box_constraint=True):
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        self.lr = lr
        self.device = device if device is not None else model.device
        self.lambda_l2 = lambda_l2
        self.lambda_stoi = lambda_stoi
        self.stoi_threshold = stoi_threshold
        self.stoi_check_interval = stoi_check_interval
        self.use_box_constraint = use_box_constraint
        
        import inspect
        sig = inspect.signature(self.processor.__call__)
        params = list(sig.parameters.keys())
        self.audio_param = 'audios' if 'audios' in params else 'audio'

    def load_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
        return audio

    def get_target_ids(self, target_text):
        text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n<|im_end|>\n<|im_start|>assistant\n"
        full_text = text_prompt + target_text
        
        inputs = self.processor.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs['input_ids'][0]
        
        assistant_token = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
        assistant_idx = (input_ids == assistant_token).nonzero(as_tuple=True)[0]
        
        if len(assistant_idx) > 0:
            start_idx = assistant_idx[-1] + 2
            target_ids = input_ids[start_idx:start_idx + len(target_text.split()) + 5]
        else:
            target_ids = input_ids[-len(target_text.split()) - 5:]
        
        return target_ids.to(self.device)

    def get_assistant_start_pos(self, input_ids):
        assistant_token = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
        assistant_idx = (input_ids == assistant_token).nonzero(as_tuple=True)[0]
        
        if len(assistant_idx) > 0:
            return min(assistant_idx[-1] + 2, input_ids.shape[0] - 1)
        return input_ids.shape[0] - 1

    def _reconstruct_waveform(self, features, reference_audio=None):
        features_np = features.squeeze().detach().cpu().numpy()
        
        if not np.all(np.isfinite(features_np)):
            features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_fft = 400
        hop_length = 160
        win_length = 400
        
        mel_basis = librosa.filters.mel(
            sr=self.processor.feature_extractor.sampling_rate, 
            n_fft=n_fft, 
            n_mels=features_np.shape[0]
        )
        eps = 1e-10
        linear_spec = np.dot(np.linalg.pinv(mel_basis), np.exp(features_np) - eps)
        linear_spec = np.maximum(linear_spec, eps)
        
        audio = librosa.griffinlim(linear_spec, n_iter=32, hop_length=hop_length, win_length=win_length)
        
        # Energy normalization if reference audio is provided
        if reference_audio is not None:
            ref_energy = np.mean(reference_audio ** 2)
            audio_energy = np.mean(audio ** 2)
            if audio_energy > 1e-10 and ref_energy > 1e-10:
                scale = np.sqrt(ref_energy / audio_energy)
                scale = np.clip(scale, 0.01, 100.0)
                audio = audio * scale
        
        return audio
    
    def _compute_stoi_check(self, original_features, perturbed_features, original_audio):
        if not STOI_AVAILABLE:
            return 1.0
        
        try:
            adv_audio = self._reconstruct_waveform(perturbed_features)
            
            # Ensure audio is finite and has sufficient length
            if not np.all(np.isfinite(adv_audio)):
                return 1.0
            
            min_len = min(len(original_audio), len(adv_audio))
            if min_len < 256:  # STOI needs minimum length
                return 1.0
            
            orig_short = original_audio[:min_len]
            adv_short = adv_audio[:min_len]
            
            # Normalize to prevent STOI issues
            orig_short = orig_short / (np.max(np.abs(orig_short)) + 1e-8)
            adv_short = adv_short / (np.max(np.abs(adv_short)) + 1e-8)
            
            sr = self.processor.feature_extractor.sampling_rate
            stoi_val = compute_stoi(orig_short, adv_short, sr, extended=False)
            return stoi_val
        except Exception as e:
            print(f"  STOI check failed: {e}, returning 1.0")
            return 1.0
    
    def _compute_delta_from_alpha(self, alpha):
        """Box constraint: delta = epsilon * tanh(alpha), ensuring |delta| <= epsilon."""
        return self.epsilon * torch.tanh(alpha)
    
    def optimize_delta_feature(self, audio_path, target_text, num_steps=500, verbose=True):
        audio = self.load_audio(audio_path)
        
        text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n<|im_end|>\n<|im_start|>assistant\n"
        
        base_inputs = self.processor(text=text_prompt, **{self.audio_param: audio}, return_tensors="pt")
        
        input_features = base_inputs['input_features'].to(self.device).clone().detach()
        input_features.requires_grad = False
        
        feature_attention_mask = base_inputs['feature_attention_mask'].to(self.device).clone().detach()
        
        text_inputs = {
            'input_ids': base_inputs['input_ids'].to(self.device),
            'attention_mask': base_inputs['attention_mask'].to(self.device)
        }
        
        target_ids = self.get_target_ids(target_text)
        assistant_start_pos = self.get_assistant_start_pos(base_inputs['input_ids'][0])
        
        print(f"Assistant start position: {assistant_start_pos}")
        print(f"Target IDs: {target_ids}")
        print(f"Target IDs decoded: {self.processor.tokenizer.decode(target_ids)}")
        print(f"Lambda L2: {self.lambda_l2}, Lambda STOI: {self.lambda_stoi}")
        print(f"Epsilon: {self.epsilon}, Box constraint: {self.use_box_constraint}")
        
        if self.lambda_stoi > 0 and not STOI_AVAILABLE:
            print("WARNING: pystoi not available, STOI loss will be ignored")
            self.lambda_stoi = 0.0
        
        if self.use_box_constraint:
            # Use tanh parameterization for strict box constraint
            alpha = torch.nn.Parameter(torch.zeros_like(input_features, device=self.device))
            optimizer = torch.optim.Adam([alpha], lr=self.lr)
        else:
            delta = torch.nn.Parameter(torch.zeros_like(input_features, device=self.device))
            optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        losses = []
        attack_losses = []
        l2_losses = []
        stoi_values = []
        
        if self.use_box_constraint:
            best_alpha = alpha.detach().clone()
        best_attack_loss = float('inf')
        
        pbar = tqdm(range(num_steps), desc="Optimizing delta") if verbose else range(num_steps)
        
        for step in pbar:
            optimizer.zero_grad()
            
            if self.use_box_constraint:
                delta = self._compute_delta_from_alpha(alpha)
            perturbed_features = input_features + delta
            perturbed_features = torch.clamp(perturbed_features, -15, 5)
            
            inputs = {
                **text_inputs,
                'input_features': perturbed_features,
                'feature_attention_mask': feature_attention_mask
            }
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            target_len = len(target_ids)
            logits_start = assistant_start_pos
            
            if logits_start + target_len > logits.shape[1]:
                target_len = logits.shape[1] - logits_start
            
            attack_loss = torch.tensor(0.0, device=self.device)
            if target_len > 0:
                attack_loss = F.cross_entropy(
                    logits[:, logits_start:logits_start + target_len, :].reshape(-1, logits.shape[-1]),
                    target_ids[:target_len].repeat(logits.shape[0])
                )
            
            l2_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_l2 > 0:
                l2_loss = torch.sum((perturbed_features - input_features) ** 2)
            
            total_loss = attack_loss + self.lambda_l2 * l2_loss
            
            total_loss.backward()
            
            optimizer.step()
            
            losses.append(total_loss.item())
            attack_losses.append(attack_loss.item())
            l2_losses.append(l2_loss.item())
            
            if self.lambda_stoi > 0 and step > 0 and step % self.stoi_check_interval == 0:
                current_delta = self._compute_delta_from_alpha(alpha).detach() if self.use_box_constraint else delta.detach()
                current_features = (input_features + current_delta).detach()
                stoi_val = self._compute_stoi_check(input_features, current_features, audio)
                stoi_values.append(stoi_val)
                
                if verbose:
                    print(f"\n  Step {step}: STOI = {stoi_val:.4f} (threshold: {self.stoi_threshold})")
                
                if stoi_val < self.stoi_threshold and stoi_val > 0.01:
                    scale_factor = max(stoi_val / self.stoi_threshold, 0.5)
                    if self.use_box_constraint:
                        with torch.no_grad():
                            alpha.data *= scale_factor
                    else:
                        with torch.no_grad():
                            delta.data *= scale_factor
                    if verbose:
                        print(f"  ** STOI below threshold, scaled by {scale_factor:.4f}")
                elif stoi_val <= 0.01:
                    if verbose:
                        print(f"  ** STOI too low ({stoi_val:.4f}), reducing perturbation")
                    if self.use_box_constraint:
                        with torch.no_grad():
                            alpha.data *= 0.5
                    else:
                        with torch.no_grad():
                            delta.data *= 0.5
            
            if attack_loss.item() < best_attack_loss:
                best_attack_loss = attack_loss.item()
                if self.use_box_constraint:
                    best_alpha = alpha.detach().clone()
                else:
                    best_delta = delta.detach().clone()
            
            if verbose and step % 50 == 0:
                postfix = {
                    "total_loss": f"{total_loss.item():.4f}",
                    "attack_loss": f"{attack_loss.item():.4f}",
                    "l2_loss": f"{l2_loss.item():.4f}" if self.lambda_l2 > 0 else "N/A"
                }
                if self.lambda_stoi > 0 and stoi_values:
                    postfix["stoi"] = f"{stoi_values[-1]:.4f}"
                pbar.set_postfix(postfix)
        
        if self.use_box_constraint:
            final_delta = self._compute_delta_from_alpha(best_alpha).detach()
        else:
            final_delta = best_delta
        jailbreak_features = input_features + final_delta
        
        result = {
            'delta': final_delta.cpu(),
            'jailbreak_features': jailbreak_features.cpu(),
            'original_features': input_features.cpu(),
            'losses': losses,
            'attack_losses': attack_losses,
            'l2_losses': l2_losses,
            'stoi_values': stoi_values,
            'target_ids': target_ids.cpu(),
            'audio': audio,
            'text_inputs': text_inputs,
            'feature_attention_mask': feature_attention_mask.cpu(),
            'assistant_start_pos': assistant_start_pos
        }
        
        if self.lambda_l2 > 0 or self.lambda_stoi > 0:
            result['config'] = {
                'lambda_l2': self.lambda_l2,
                'lambda_stoi': self.lambda_stoi,
                'stoi_threshold': self.stoi_threshold,
                'stoi_check_interval': self.stoi_check_interval
            }
        
        return result

    def test_attack_with_features(self, input_features, feature_attention_mask, text_inputs):
        inputs = {
            **text_inputs,
            'input_features': input_features.to(self.device),
            'feature_attention_mask': feature_attention_mask.to(self.device)
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        response = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return response

    def test_attack(self, audio, prompt="Please listen to the audio and answer the spoken request."):
        text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.processor(text=text_prompt, **{self.audio_param: audio}, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        response = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return response

    def save_audio(self, audio, output_path):
        sr = self.processor.feature_extractor.sampling_rate
        sf.write(output_path, audio, sr)
    
    def save_adversarial_audio(self, audio_path, delta, output_path):
        audio = self.load_audio(audio_path)
        
        text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n<|im_start|>assistant\n"
        inputs = self.processor(text=text_prompt, **{self.audio_param: audio}, return_tensors="pt")
        input_features = inputs['input_features']
        
        adv_features = input_features + delta
        adv_features = torch.clamp(adv_features, -15, 5)
        adv_features_np = adv_features.squeeze().numpy()
        
        # Handle NaN/Inf
        if not np.all(np.isfinite(adv_features_np)):
            adv_features_np = np.nan_to_num(adv_features_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_fft = 400
        hop_length = 160
        win_length = 400
        
        mel_basis = librosa.filters.mel(
            sr=self.processor.feature_extractor.sampling_rate, 
            n_fft=n_fft, 
            n_mels=adv_features_np.shape[0]
        )
        eps = 1e-10
        linear_spec = np.dot(np.linalg.pinv(mel_basis), np.exp(adv_features_np) - eps)
        linear_spec = np.maximum(linear_spec, eps)
        
        adv_audio = librosa.griffinlim(linear_spec, n_iter=32, hop_length=hop_length, win_length=win_length)
        
        # Final safety check
        if not np.all(np.isfinite(adv_audio)):
            adv_audio = np.nan_to_num(adv_audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Energy normalization: match original audio energy to prevent explosion
        orig_energy = np.mean(audio ** 2)
        adv_energy = np.mean(adv_audio ** 2)
        if adv_energy > 1e-10 and orig_energy > 1e-10:
            scale = np.sqrt(orig_energy / adv_energy)
            # Limit scaling to prevent silence
            scale = np.clip(scale, 0.01, 100.0)
            adv_audio = adv_audio * scale
        
        sf.write(output_path, adv_audio, self.processor.feature_extractor.sampling_rate)
        return output_path