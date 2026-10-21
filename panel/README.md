# LLM Control Panel

หน้าเว็บเดียวสำหรับวงจรเทรน LLM: **ทำความสะอาดข้อมูล → วัด perplexity → เทรนด้วย Ludwig → ประเมิน → เทียบผล**
ครอบ repo `data-pipeline` และ `evaluation` เดิมโดยไม่ต้องแก้ core logic ของมัน
ทุกอย่างรันบนเครื่องคุณ ข้อมูลไม่ออกนอกเครื่อง

## โครงสร้างโฟลเดอร์ที่แนะนำ

```
ainrg-llm-dev/
├── data-pipeline/      (repo เดิม)
├── evaluation/         (repo เดิม)
└── llm-panel/          (โฟลเดอร์นี้)
```

ถ้าวาง repo ไว้ที่อื่น ตั้ง path ได้ที่หน้า **Settings**

## ติดตั้ง (macOS)

```bash
cd llm-panel
./run_panel.sh            # ครั้งแรกจะสร้าง .venv และติดตั้ง dependency ของ panel ให้
```

panel เองเบามาก (streamlit, pandas, pyyaml) ส่วน torch / transformers / Ludwig อยู่ใน env ของแต่ละเครื่องมือ
แล้วบอก panel ว่าแต่ละขั้นใช้ python ตัวไหนที่หน้า **Settings** เช่น

| ขั้น | ต้องมีอะไรใน env นั้น |
|---|---|
| data | `pip install -r ../data-pipeline/requirements.txt` + Thai-Data-Privacy (ดู README ของ data-pipeline) |
| perplexity | torch, transformers (ใช้ env เดียวกับ data ได้) |
| eval | `pip install -r ../evaluation/requirements.txt` |
| train | `pip install "ludwig[llm]"` (ต้อง **Python 3.12 ขึ้นไป**) |

Ludwig ไม่จำเป็นต้อง clone มาเอง ติดตั้งจาก pip ก็พอ ยกเว้นอยากแก้ source ของ Ludwig
(ถ้า clone มา ให้ `pip install -e "./ludwig[llm]"` ใน env ของขั้น train)

กดปุ่ม **ตรวจ Python ทั้งหมด** ในหน้า Settings เพื่อเช็คว่าแต่ละ env import ได้ครบ

## ใช้งาน

1. **เลือก experiment** ที่แถบซ้าย (ไฟล์ `experiments/<ชื่อ>.yaml`) ค่าทุกหน้าผูกกับ experiment นี้
   สร้างใหม่หรือคัดลอกจากของเดิมได้ เหมาะกับการลองหลายสูตร
2. **Data Cleaning** ตั้งค่า pipeline เป็นกลุ่ม ๆ (PDPA, Thai NER, ภาษาไทย, fastText) แทนการพิมพ์ argument ยาว ๆ
   ตั้ง `debug_first_n` เพื่อลองกับข้อมูลไม่กี่แถวก่อน ผลลัพธ์มีสรุปตัวเลข, ธงที่ตรวจพบ, ตัวอย่าง GOOD/BAD
3. **Perplexity** เลือกไฟล์จากผล Data Cleaning ได้เลย ผลลัพธ์มี
   - histogram ของ PPL รายข้อความ
   - กราฟ **chars/token เทียบ PPL** ใช้แยกปัญหา tokenizer/vocab (ซ้าย) ออกจากปัญหาเนื้อหา (ขวา)
   - รายการข้อความที่ PPL สูง/ต่ำผิดปกติ พร้อมเปิดดูข้อความจริง
   - โทเค็นที่โมเดลสะดุดบ่อย พร้อมบริบท
4. **Train (Ludwig)** กรอกฟอร์ม (base model, คอลัมน์, prompt template, LoRA, epochs ...) panel สร้าง config ให้
   หรือเปิดโหมดแก้ YAML เอง เมื่อเทรนเสร็จจะ export เป็นโฟลเดอร์ Hugging Face ให้อัตโนมัติ แล้วกด **ประเมินโมเดลนี้** ได้ทันที
5. **Evaluate** เลือกโมเดล (HF / vLLM / OpenAI), โหมด pt/it, ชุดข้อสอบ ผลลัพธ์มีคะแนนราย slice (subject, ปี, ระดับ),
   **confusion matrix** (เห็นว่าโมเดลเอียงไปตอบตัวเลือกไหน หรือดึงคำตอบไม่ได้บ่อยแค่ไหน) และรายการข้อที่ตอบผิด
6. **Compare models** เทียบหลายรอบ ทั้งคะแนนรวม ส่วนต่างเทียบรอบอ้างอิง และราย slice
   เพิ่มโฟลเดอร์ `results/` เก่าของ evaluation ได้ที่หน้า Settings เพื่อเทียบกับผลที่เคยรันไว้

## การรันงาน

- ทุกงานรันเป็น background process แยกจากหน้าเว็บ ปิดแท็บแล้วงานยังรันต่อ
- แต่ละงานมีโฟลเดอร์ `runs/<ขั้น>/<เวลา>_<ชื่อ>/` เก็บ `job.json` (คำสั่ง + snapshot ค่าที่ใช้), `log.txt` และผลลัพธ์
  จึงย้อนดูได้เสมอว่ารอบไหนใช้ค่าอะไร
- กด **หยุด** แล้วจะหยุดทั้ง process tree (รวม worker ของ multiprocessing)
- API key ไม่ถูกบันทึกลง experiment, `job.json` หรือ log (แสดงเป็น `***`)
  สำหรับ OpenAI เว้นช่องว่างไว้แล้วตั้ง env `OPENAI_API_KEY` ก่อนเปิด panel ได้

## เพิ่ม argument ใหม่ของเครื่องมือ

ทุก flag ถูกประกาศไว้ที่เดียวใน `core/schema.py` เช่น

```python
P("chunk_size", "--chunk-size", "int", 50000, "บรรทัดต่อชิ้นงาน", "ประสิทธิภาพ"),
```

เพิ่มหนึ่งบรรทัด ฟอร์ม, การบันทึกลง experiment และการสร้างคำสั่งจะได้มาเองทั้งหมด

## ทดสอบ

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests
```

## ข้อจำกัดที่ควรรู้

- Apple Silicon ไม่มี CUDA: quantization 4/8-bit (bitsandbytes) ใช้ไม่ได้ ให้ตั้ง `quantization_bits = none`
- ข้อมูลเทรน LLM ของ Ludwig ต้องเป็นคู่ input/output ไฟล์จาก Data Cleaning ที่มีแค่ `text` ต้องแปลงก่อน
- ขั้น eval ดาวน์โหลดชุดข้อสอบลง `evaluation/data/` ในการรันครั้งแรก
