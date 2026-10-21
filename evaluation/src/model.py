from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import torch
from typing import List 
from abc import ABC, abstractmethod
import openai
from concurrent.futures import ThreadPoolExecutor
from .prompt_process.formatters import FORMATTERS
from typing import List
import requests
def _normalize_thinking(thinking_mode):
    """แปลงค่า thinking ให้เป็นรูปแบบเดียว: "True" | "False" | "no_chat_template" | None"""
    if thinking_mode is True:
        return "True"
    if thinking_mode is False:
        return "False"
    if isinstance(thinking_mode, str):
        if thinking_mode.lower() == "true":
            return "True"
        if thinking_mode.lower() == "false":
            return "False"
        if thinking_mode == "no_chat_template":
            return thinking_mode
    return None


class BaseModel(ABC):
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

class HFModel(BaseModel):
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            pad_token_id=self.tokenizer.pad_token_id,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        ).eval()
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(
                self.checkpoint_path,
                pad_token_id=self.tokenizer.pad_token_id,
                trust_remote_code=True
            )
        except OSError:
            pass  # บาง checkpoint ไม่มี generation_config.json
        # decoder-only ต้อง pad ด้านซ้าย ไม่งั้นตำแหน่งสุดท้าย (logits[:, -1]) และการ generate
        # ของตัวอย่างที่สั้นกว่าใน batch จะไปตกที่ pad token
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_logits(self, inputs: List[str], max_seq_len: int = 8192):
        """คืน logits ดิบของโทเค็นถัดไป (ยังไม่ softmax) ขนาด [batch, vocab]

        ใช้ left padding + attention_mask จาก tokenizer + position_ids ที่ไม่นับ pad
        จึงใช้ batch size มากกว่า 1 ได้ในโหมด pt
        """
        enc = self.tokenizer(inputs, padding='longest', return_tensors='pt')
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
            attention_mask = attention_mask[:, -max_seq_len:]

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        try:
            outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        except TypeError:
            outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"][:, -1, :].float()
        return logits, {"tokens": {"input_ids": input_ids}}

    def format_instructions(self, prompt: str, thinking_mode=None) -> str:
        """thinking_mode: "True" | "False" | "no_chat_template" | None (รับ bool ได้ด้วย)

        เดิม evaluator ส่ง bool มาแต่ฟังก์ชันเช็คเป็น string ทำให้ตกไป branch ที่ไม่มี
        generation prompt และ thinking flag ไม่มีผล
        """
        mode = _normalize_thinking(thinking_mode)
        messages = [
            {"role": "user", "content": f"{prompt} \nช่วยตอบลงในช่อง \\boxed{{}} เท่านั้นหากตอบถูกแต่ไม่อยู่ในรูปแบบนี้จะไม่ถูกนับเป็นคำตอบที่ถูกต้อง"},
        ]
        if mode == "no_chat_template":
            return messages[0]['content'] + "The answer is: "
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if mode in ("True", "False"):
            kwargs["enable_thinking"] = (mode == "True")
        return self.tokenizer.apply_chat_template(messages, **kwargs)


class OpenAIModel(BaseModel):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def generate_prompt(self, prompt, **kwargs):
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                **kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"OpenAI request failed: {e}")
            return ""

    def generate(self, prompts, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda p: self.generate_prompt(p, **kwargs), prompts))
        return results

    def format_instructions(self, prompt: str, choices: List[str] = None, thinking_mode=None):
        # choices เป็น optional: evaluator ไม่ได้ส่งมา (เดิมทำให้โหมด it กับ API error ทันที)
        messages = [
            {"role": "user", "content": f"{prompt} \nช่วยตอบลงในช่อง \\boxed{{}} เท่านั้นหากตอบถูกแต่ไม่อยู่ในรูปแบบนี้จะไม่ถูกนับเป็นคำตอบที่ถูกต้อง"},
        ]
        return messages
    def load_model(self):
        pass


class VLLMModel(BaseModel):
    def __init__(self, base_url: str, model_name: str, api_key: str = None):
        """
        Initialize VLLM Model
        
        Args:
            base_url: Base URL of vLLM API server (e.g., "http://example/thaillm-inference")
            model_name: Model name (e.g., "thaillm-8b")
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        self.model_name = model_name
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def load_model(self):
        """vLLM API doesn't need to load model locally"""
        pass
    
    def generate_prompt(self, prompt, **kwargs):
        """
        Generate response for single prompt
        
        Args:
            prompt: List of messages or string prompt
            **kwargs: Additional parameters like temperature, top_p, etc.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be string or list of messages")
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Unexpected response format: {e}")
            return None
    
    def generate(self, prompts, max_workers: int = 5, **kwargs):
        """
        Generate responses for multiple prompts with parallel processing
        
        Args:
            prompts: Single prompt or list of prompts
            max_workers: Number of parallel workers
            **kwargs: Additional parameters like temperature, top_p, etc.
        """
        if isinstance(prompts, (str, list)) and not isinstance(prompts[0] if isinstance(prompts, list) and prompts else None, dict):
            if isinstance(prompts, str):
                prompts = [prompts]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda p: self.generate_prompt(p, **kwargs), 
                prompts
            ))
        
        return results
    
    def generate_with_completions_api(self, prompt: str, **kwargs):
        """
        Alternative method using completions API (if available)
        
        Args:
            prompt: String prompt
            **kwargs: Additional parameters
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                headers=self.headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def format_instructions(self, prompt: str, choices: List[str] = None, thinking_mode=None):
        """
        Format instructions for multiple choice questions
        
        Args:
            prompt: Question prompt
            choices: List of answer choices (ไม่ได้ใช้ คงไว้เพื่อความเข้ากันได้)
            thinking_mode: ส่งต่อไปเป็น chat_template_kwargs.enable_thinking ตอน generate
        """
        messages = [
            {"role": "user", "content": f"{prompt} \nช่วยตอบลงในช่อง \\boxed{{}} เท่านั้นหากตอบถูกแต่ไม่อยู่ในรูปแบบนี้จะไม่ถูกนับเป็นคำตอบที่ถูกต้อง"},
        ]
        return messages
    
    def get_model_info(self):
        """
        Get information about available models
        """
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get model info: {e}")
            return None
    
    def test_connection(self):
        """
        Test connection to vLLM API server
        """
        # generate_prompt คืน None เมื่อเรียกไม่สำเร็จ (เดิมคืน True เสมอเพราะไม่ได้เช็คค่านี้)
        try:
            test_response = self.generate_prompt("Hello", max_tokens=10)
            return test_response is not None
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False