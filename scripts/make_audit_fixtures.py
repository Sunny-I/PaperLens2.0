"""Recreate small ground-truth PDFs for functional tests; no source papers are modified."""
from pathlib import Path
import json
import pymupdf

ROOT = Path(__file__).resolve().parents[1] / 'audit' / 'fixtures'

def write_pdf(name, pages, **options):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page()
            if page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), text, fontsize=11) < 0:
                raise ValueError('Fixture text did not fit')
        path.write_bytes(doc.tobytes(**options))
    return path

def main():
    write_pdf('valid/orion.pdf', [
        'Orion Sensor Study\nAbstract: Orion investigates whether data augmentation improves sensor classification.\nMethods: Orion adds small Gaussian perturbations to training measurements only. The dataset contains 120 training samples and 40 test samples.',
        'Orion Sensor Study\nResults: Orion correctly classified 34 of 40 test samples, achieving 85% accuracy. The baseline correctly classified 28 of 40.\nLimitations: Orion was evaluated on one synthetic dataset. Real clinical data were not evaluated.'
    ])
    write_pdf('valid/lyra.pdf', [
        'Lyra Sensor Study\nAbstract: Lyra investigates smoothing for sensor classification.\nMethods: Lyra uses a moving average filter and does not create new training examples. The dataset contains 120 training samples and 40 test samples.',
        'Lyra Sensor Study\nResults: Lyra correctly classified 30 of 40 test samples, achieving 75% accuracy.\nLimitations: Smoothing may remove short events.'
    ])
    write_pdf('conflict.pdf', ['Orion Replication\nResults: In the replication, Orion achieved 70% accuracy rather than the original 85%. The replication used a different dataset.'])
    write_pdf('long-31-pages.pdf', [f'Long Study Page {i}\nMethods: Page {i} describes attention measurements. ' + ('Training examples remain separate from test examples. ' * 40) for i in range(1, 32)])
    write_pdf('blank.pdf', [''])
    write_pdf('encrypted.pdf', ['Protected data.'], encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw='audit-owner', user_pw='audit-password')
    write_pdf('unicode.pdf', ['Unicode Study\nMethods: café naïve résumé - accuracy 85% ± 2%.'])
    (ROOT / 'empty.pdf').write_bytes(b'')
    (ROOT / 'corrupt.pdf').write_bytes(b'not a PDF')
    with pymupdf.open() as doc:
        p=doc.new_page()
        pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
        pix.clear_with(255)
        p.insert_image(pymupdf.Rect(50,50,150,150),pixmap=pix)
        (ROOT / 'scanned.pdf').write_bytes(doc.tobytes())
    labels=[{'query':'Orion accuracy test samples', 'relevant':[{'filename':'orion.pdf','page':2}]},
            {'query':'Lyra moving average filter', 'relevant':[{'filename':'lyra.pdf','page':1}]},
            {'query':'xylophonexyzzy', 'relevant':[]}]
    (ROOT / 'queries.json').write_text(json.dumps(labels,indent=2))
    print('Created reproducible fixtures under audit/fixtures')

if __name__=='__main__':
    main()
