# Trainer-Pipeline

ชุดเครื่องมือเทรน LLM ภาษาไทยแบบครบวงจร พร้อม control panel บนเว็บที่รันบนเครื่องตัวเอง

```
ทำความสะอาดข้อมูล → วัด perplexity → เทรน (Ludwig) → ประเมินด้วยข้อสอบไทย → เทียบผลหลายโมเดล
```

| โฟลเดอร์ | คืออะไร |
|---|---|
| [`data-pipeline/`](data-pipeline/) | ทำความสะอาดข้อความไทย: sanitize, PDPA, Thai NER, แก้สระลอย, คัดกรองด้วย fastText และวัด perplexity |
| [`evaluation/`](evaluation/) | ประเมินโมเดลด้วยข้อสอบไทย 9 ชุด (belebele, xcopa, xnli, mmlu, mmlu_thai, mmlu_proX_thai, m3exam, m6exam, thai_exam) |
| [`panel/`](panel/) | หน้าเว็บ Streamlit ที่ครอบทุกขั้นไว้ที่เดียว แทนการพิมพ์ argument ยาว ๆ และสลับหลายเครื่องมือ |

`data-pipeline/` และ `evaluation/` นำเข้าจาก [ainrg-llm-dev/Data-pipeline](https://github.com/ainrg-llm-dev/Data-pipeline)
และ [ainrg-llm-dev/evaluation](https://github.com/ainrg-llm-dev/evaluation) พร้อมประวัติ commit เดิม แล้วแก้บั๊กต่อจากนั้น

## เริ่มใช้งาน (macOS / Linux)

```bash
git clone https://github.com/ThanadolSaelem/Trainer-Pipeline.git
cd Trainer-Pipeline/panel
./run_panel.sh
```

panel ใช้ dependency เบา ๆ ของตัวเอง ส่วนงาน ML รันด้วย Python env ที่คุณกำหนดในหน้า **Settings**
ตัวอย่างการเตรียม env

```bash
# env สำหรับ data / perplexity / eval
python3 -m venv ~/venvs/trainer
~/venvs/trainer/bin/pip install -r data-pipeline/requirements.txt -r evaluation/requirements.txt
~/venvs/trainer/bin/pip install beautifulsoup4 https://github.com/PyThaiNLP/Thai-Data-Privacy/archive/master.zip

# env สำหรับเทรน (Ludwig ต้อง Python 3.12+)
python3.12 -m venv ~/venvs/ludwig
~/venvs/ludwig/bin/pip install "ludwig[llm]"
```

จากนั้นในหน้า Settings ตั้ง python ของขั้น data, perplexity, eval เป็น `~/venvs/trainer/bin/python`
และขั้น train เป็น `~/venvs/ludwig/bin/python` แล้วกด **ตรวจ Python ทั้งหมด**

โมเดลที่ data-pipeline ต้องใช้ (fastText, Thai NER) ดาวน์โหลดตาม [data-pipeline/README.md](data-pipeline/README.md)

## การแก้บั๊กหลักจาก repo ต้นทาง

**data-pipeline**
- ไฟล์ BAD มีอักขระพินทุติดหน้าชื่อ, `--single-process` เขียนข้อมูลซ้ำ, `--debug-first-n` ทำให้ resume ข้ามข้อมูล
- ตัวเลือกที่อยู่ (`--addr-*`) และการข้ามชื่อบุคคลสาธารณะไม่เคยทำงาน, ขั้น `emoji` ในค่าเริ่มต้นไม่เคยลบอีโมจิ
- Perplexity: โทเค็นช่วง overlap ถูกนับซ้ำ, ข้อความสั้นถูก pad ยาวเต็ม context, `PPL_micro` คำนวณผิดสูตร
- เพิ่ม `--per_text_jsonl` / `--top_k_tokens` สำหรับวิเคราะห์รายข้อความและโทเค็นที่โมเดลสะดุด

**evaluation**
- โหมด `it`: thinking flag ไม่มีผลและไม่มี generation prompt, โหมด API error ทุกครั้ง
- โหมด `pt`: อ่าน logit ที่ตำแหน่ง pad ทำให้ต้องใช้ batch 1 ตอนนี้ใช้ batch ใหญ่ได้
- ดึงคำตอบจาก `\boxed{}` ตัวสุดท้ายนอกส่วน think, ค่าเริ่มต้นเป็น greedy เพื่อให้ผลซ้ำได้ (`--do-sample` เพื่อสุ่ม)

รายละเอียดอยู่ใน commit message ของแต่ละ commit

## ทดสอบ

```bash
(cd data-pipeline && python -m pytest tests -q)
(cd evaluation && python -m pytest tests -q)
(cd panel && .venv/bin/python -m pytest tests -q)
```

test ของ data-pipeline และ evaluation สร้างโมเดล Llama ขนาดจิ๋วขึ้นมาเอง ไม่ต้องดาวน์โหลดอะไร
