from pathlib import Path
import sys

from pypdf import PdfReader, PdfWriter


src = Path(sys.argv[1]).resolve()
tmp = src.with_name(src.stem + "_metadata.pdf")
reader = PdfReader(src)
writer = PdfWriter()
for page in reader.pages:
    writer.add_page(page)
writer.add_metadata(
    {
        "/Title": "DeepBlock：拓扑共设计的量子局部搜索——Baihua真机上的CVRP闭环证据",
        "/Author": "参赛团队（姓名、单位待补）",
        "/Subject": "量子+优化赛道半决赛论文",
        "/Keywords": "CVRP; QAOA; QUBO; Baihua; hybrid quantum-classical optimization",
    }
)
with tmp.open("wb") as stream:
    writer.write(stream)
tmp.replace(src)
print(src)
