import asyncio
import re
import subprocess
import fitz
import os
from typing import List, Dict, Any, Tuple

async def run_nougat_on_pdf(pdf_path: str, output_dir: str) -> str | None:
    """Runs Nougat on a PDF (with reduced batch size) and returns the path to the output .mmd file."""
    os.makedirs(output_dir, exist_ok=True)
    output_md_path = os.path.join(output_dir, os.path.basename(pdf_path).replace('.pdf', '.mmd'))
    try:
        print(f"Running Nougat on {pdf_path} (batch size 1)...")
        command = ["nougat", "--batchsize", "1", pdf_path, "-o", output_dir]
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        await asyncio.sleep(0.5)
        if process.returncode == 0:
            if os.path.exists(output_md_path):
                print(f"Nougat finished for {pdf_path}")
                return output_md_path
            else:
                print(f"ERROR: Nougat ran but output file missing: {output_md_path}\nStderr: {stderr.decode()}")
                return None
        else:
            print(f"ERROR: Nougat failed (Code {process.returncode}): {stderr.decode()}")
            return None
    except FileNotFoundError:
        print("ERROR: 'nougat' command not found.")
        return None
    except Exception as e:
        print(f"ERROR: Exception running Nougat: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_images_and_regions(pdf_path: str, output_dir: str) -> Dict[int, List[Dict]]:
    """Extracts images and potential table/figure regions per page."""
    os.makedirs(output_dir, exist_ok=True)
    images_metadata = {}
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            images_metadata[page_num + 1] = []
            img_list = page.get_images(full=True)
            for img_index, img in enumerate(img_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    img_filename = f"{os.path.basename(pdf_path)}_p{page_num+1}_img{img_index}.{ext}"
                    img_path = os.path.join(output_dir, img_filename)
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    img_bboxes = page.get_image_bbox(img, transform=False)
                    bbox = tuple(img_bboxes) if isinstance(img_bboxes, fitz.Rect) else tuple(img_bboxes[0]) if img_bboxes else None
                    images_metadata[page_num + 1].append({'path': img_path, 'bbox': bbox, 'type': 'image'})
                except Exception as img_extract_err:
                    print(f"Warn: Failed extract img xref {xref} pg {page_num+1}: {img_extract_err}")
        doc.close()
    except Exception as e:
        print(f"ERROR: Failed processing PDF images {pdf_path}: {e}")
    return images_metadata

async def get_llava_description(image_path: str) -> str | None:
    """Gets description for an image using LLaVA (Placeholder)."""
    await asyncio.sleep(0.05)
    if not os.path.exists(image_path):
        return None
    if any(kw in os.path.basename(image_path).lower() for kw in ["table", "chart", "graph", "figure", "plot", "img", "diagram", "schema"]):
        return f"Placeholder LLaVA: Detailed analysis of visual element {os.path.basename(image_path)}."
    return None

def parse_nougat_markdown(md_content: str, image_analysis: Dict[int, List[Dict]], filename: str) -> List[Tuple[str, Dict]]:
    """Parses Nougat Markdown, extracts text chunks, LaTeX, and associates LLaVA data."""
    chunks_with_metadata = []
    paragraphs = re.split(r'\n{2,}', md_content)
    current_page = 1
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        page_match = re.match(r'^#+\s*Page\s*(\d+)', para, re.IGNORECASE)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        if len(para) < 40 and (re.match(r'^\d+$', para) or para.lower() == filename.lower()):
            continue
        metadata = {
            "file_name": filename,
            "page_number": current_page,
            "contains_math": False,
            "latex_formulas": [],
            "llava_descriptions": [],
            "source_type": "text"
        }
        latex_blocks = re.findall(r'(\$\$.*?\$\$|\$.*?\$|\\begin\{.*?\}[\s\S]*?\\end\{.*?\}|\\\[.*?\\\]|\\\(.*?\\\))', para, re.DOTALL)
        if latex_blocks:
            metadata["contains_math"] = True
            metadata["latex_formulas"] = [block.strip() for block in latex_blocks]
        text_for_embedding = para
        page_llava_desc = []
        if current_page in image_analysis:
            for item in image_analysis[current_page]:
                if item.get('description'):
                    page_llava_desc.append(item['description'])
        if page_llava_desc:
            metadata["llava_descriptions"] = page_llava_desc
            metadata["source_type"] = "text_with_visual"
        if text_for_embedding:
            chunks_with_metadata.append((text_for_embedding, metadata))
    return chunks_with_metadata

def remove_think_blocks(text: str) -> str:
    """Removes <think>...</think> blocks from a string."""
    if not text:
        return ""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()