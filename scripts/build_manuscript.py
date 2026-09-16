"""Build editable Word equations with Pandoc, then format the Russian manuscript.

Requires pandoc and python-docx. Render and inspect the output before submission.
"""
import argparse
from pathlib import Path
import subprocess
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]/"paper"/"article_ru.md"
    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["pandoc",str(source),"--from=markdown+tex_math_dollars","--standalone","-o",str(out)],check=True)
    doc=Document(out)
    sec=doc.sections[0]
    sec.page_width=Inches(8.5); sec.page_height=Inches(11)
    sec.top_margin=Inches(0.65); sec.bottom_margin=Inches(0.65)
    sec.left_margin=Inches(0.8); sec.right_margin=Inches(0.8)
    for s in doc.styles:
        if s.type==1:
            s.font.name="Times New Roman"
            fonts=s.element.get_or_add_rPr().get_or_add_rFonts()
            for attr in list(fonts.attrib):
                if attr.endswith("Theme"): del fonts.attrib[attr]
            for attr in ("ascii","hAnsi","eastAsia","cs"):
                fonts.set(qn(f"w:{attr}"),"Times New Roman")
            s.font.size=Pt(11.5)
            s.font.color.rgb=RGBColor(0,0,0)
            s.paragraph_format.line_spacing=1.05
            s.paragraph_format.space_after=Pt(5)
    for name in ("Normal","Body Text","First Paragraph"):
        s=doc.styles[name]
        s.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        s.paragraph_format.first_line_indent=Inches(0.22)
        s.paragraph_format.widow_control=True
    title=doc.styles["Title"]
    title.font.size=Pt(16); title.font.bold=True
    title.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after=Pt(6)
    title.paragraph_format.first_line_indent=Inches(0)
    for name in ("Author","Subtitle"):
        if name in doc.styles:
            doc.styles[name].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
            doc.styles[name].paragraph_format.first_line_indent=Inches(0)
    for i in range(1,4):
        s=next(style for style in doc.styles if style.style_id==f"Heading{i}")
        s.font.size=Pt(12); s.font.bold=True
        s.paragraph_format.space_before=Pt(10)
        s.paragraph_format.space_after=Pt(5)
        s.paragraph_format.keep_with_next=True
        s.paragraph_format.first_line_indent=Inches(0)
    paragraphs=doc.paragraphs
    for position,p in enumerate(paragraphs):
        if position+1 < len(paragraphs) and paragraphs[position+1]._p.xpath('.//m:oMathPara'):
            p.paragraph_format.keep_with_next=True
        if p.style.style_id.startswith("Heading") or p.style.name=="Title":
            for r in p.runs:
                r.font.name="Times New Roman"
                r.bold=True
                fonts=r._r.get_or_add_rPr().get_or_add_rFonts()
                for attr in list(fonts.attrib):
                    if attr.endswith("Theme"): del fonts.attrib[attr]
                fonts.set(qn("w:ascii"),"Times New Roman")
                fonts.set(qn("w:hAnsi"),"Times New Roman")
        if p._p.xpath('.//m:oMathPara'):
            p.paragraph_format.first_line_indent=Inches(0)
            p.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before=Pt(3)
            p.paragraph_format.space_after=Pt(6)
        if p.text.startswith(("Аннотация.","Ключевые слова:")):
            p.paragraph_format.first_line_indent=Inches(0)
        if p.text=="Литература":
            p.paragraph_format.page_break_before=True
        if p.style.name=="Compact":
            p.paragraph_format.first_line_indent=Inches(0)
            p.paragraph_format.space_after=Pt(4)
            for r in p.runs: r.font.size=Pt(10.5)
    for table in doc.tables:
        caption=OxmlElement("w:p")
        cp=OxmlElement("w:pPr")
        keep=OxmlElement("w:keepNext");cp.append(keep);caption.append(cp)
        run=OxmlElement("w:r");text=OxmlElement("w:t")
        text.text="Таблица 1 Сравниваемые варианты интерполяции"
        run.append(text);caption.append(run)
        table._tbl.addprevious(caption)
        table.alignment=WD_TABLE_ALIGNMENT.CENTER
        table.autofit=False
        widths=[1.25,3.15,2.45]
        for i,w in enumerate(widths): table.columns[i].width=Inches(w)
        props=table._tbl.tblPr
        borders=OxmlElement("w:tblBorders")
        for edge in ("top","left","bottom","right","insideH","insideV"):
            e=OxmlElement(f"w:{edge}"); e.set(qn("w:val"),"single"); e.set(qn("w:sz"),"4"); e.set(qn("w:color"),"D9D9D9"); borders.append(e)
        props.append(borders)
        for ri,row in enumerate(table.rows):
            trpr=row._tr.get_or_add_trPr()
            cant=OxmlElement("w:cantSplit"); trpr.append(cant)
            if ri==0: trpr.append(OxmlElement("w:tblHeader"))
            for ci,cell in enumerate(row.cells):
                cell.width=Inches(widths[ci])
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=cell._tc.get_or_add_tcPr()
                for old in list(tcpr.findall(qn("w:tcBorders"))):tcpr.remove(old)
                cb=OxmlElement("w:tcBorders")
                for edge in ("top","left","bottom","right"):
                    e=OxmlElement(f"w:{edge}");e.set(qn("w:val"),"single");e.set(qn("w:sz"),"4");e.set(qn("w:color"),"D9D9D9");cb.append(e)
                tcpr.append(cb)
                margins=OxmlElement("w:tcMar")
                for edge in ("top","left","bottom","right"):
                    m=OxmlElement(f"w:{edge}"); m.set(qn("w:w"),"70"); m.set(qn("w:type"),"dxa"); margins.append(m)
                tcpr.append(margins)
                if ri==0:
                    shade=OxmlElement("w:shd"); shade.set(qn("w:fill"),"ECECEC"); tcpr.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent=Inches(0)
                    p.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT
                    p.paragraph_format.space_after=Pt(1)
                    p.paragraph_format.line_spacing=1
                    for r in p.runs:
                        r.font.size=Pt(10.5)
                        if ri==0:r.bold=True
    # Remove inherited decorative borders from title and paragraph styles.
    for element in doc.styles.element.xpath('.//w:pBdr'):
        element.getparent().remove(element)
    footer=sec.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement("w:fldSimple"); field.set(qn("w:instr"),"PAGE"); footer._p.append(field)
    doc.core_properties.author="Салман Али"
    doc.core_properties.title="Влияние согласованности оптического потока на качество интерполяции видеокадров"
    doc.core_properties.subject="Методика и протокол экспериментальной проверки"
    doc.save(out)
    print(out)


if __name__=="__main__":main()
