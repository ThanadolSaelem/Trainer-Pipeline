import re

class PostProcessor:
    def __init__(self, text):
        # API คืน None ได้เมื่อ request ล้มเหลว
        self.text = text or ""
    
    def extract_think(self):
        think_match = re.search(r'<think>(.*?)</think>', self.text, re.DOTALL)
        return think_match.group(1).strip() if think_match else None
    
    def extract_answer(self):
        answer_match = re.search(r'<answer>(.*?)</answer>', self.text, re.DOTALL)
        return answer_match.group(1).strip() if answer_match else None
    
    def _answer_region(self):
        # ตัดส่วน <think>...</think> ออก (รวมกรณีที่ไม่มีแท็กเปิด) เพื่อไม่ให้หยิบ \boxed ที่ระหว่างคิด
        text = self.text or ""
        if "</think>" in text:
            text = text.split("</think>")[-1]
        return text

    def extract_boxed_number(self):
        """ดึงคำตอบจาก boxed{} ตัวสุดท้ายนอกส่วน think แล้ว normalize เช่น 'A. ข้อความ' -> 'A'"""
        matches = re.findall(r'\\?boxed\{([^}]+)\}', self._answer_region())
        if not matches:
            return None
        ans = matches[-1].strip()
        # \boxed{\text{A}}: regex หยุดที่ } ตัวแรก จึงเหลือ "\text{A" ให้ตัดคำสั่ง \text{ ทิ้ง
        ans = re.sub(r'\\text\{', '', ans).strip().strip('}').strip()
        first = re.split(r'[\s.)\:]', ans, maxsplit=1)[0]
        return first or ans
    
    def output(self):
        return self.text.strip()

