from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(use_angle_cls=True, lang='en',
               det_model_dir="../paddleOCR/en_PP-OCRv3_det_infer",
               rec_model_dir="../paddleOCR/en_PP-OCRv3_rec_infer",
               cls_model_dir="../paddleOCR/ch_ppocr_mobile_v2.0_cls_infer")

img_path = './custom_feed/labels/001_i_s.png'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)
