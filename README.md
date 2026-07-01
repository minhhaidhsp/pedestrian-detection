# FA-PromptDETR: Multispectral Pedestrian Detection

Tái tạo (reproduction) bài báo khoa học **"FA-PromptDETR: A Robust Multispectral
Pedestrian Detection Framework Based on DETR with Frequency-Aware Visual Prompting
and Weight-Space Ensembling"**.

Dự án xây dựng lại kiến trúc DETR cho bài toán phát hiện người đi bộ đa phổ
(RGB + thermal), với hai đóng góp chính của bài báo:

- **FA-VP (Frequency-Aware Visual Prompting)**: prompting trong miền tần số để
  tăng độ bền vững trước nhiễu/biến dạng ảnh.
- **WiSE-OD (Weight-Space Ensembling)**: nội suy trọng số giữa mô hình fine-tune
  và mô hình gốc để cân bằng hiệu năng in-distribution và out-of-distribution.

## Cài đặt

**Yêu cầu: Python 3.11.x** (dự án được pin ở bản này để đảm bảo tương thích
thư viện — không dùng 3.12/3.13/3.14).

```bash
# Windows (dùng py launcher để chọn đúng bản 3.11)
py -3.11 -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> **Lưu ý (Windows):** `pycocotools` và `torch` cần wheel tương thích với phiên
> bản Python đang dùng. Với Python 3.11, các package trong requirements.txt đều
> có wheel dựng sẵn (không cần Visual C++ Build Tools). Nếu pip không tìm được
> wheel dựng sẵn, `pycocotools` sẽ cố build từ source; có thể thử
> `pip install pycocotools-windows` thay thế.

## Cấu trúc thư mục

```
configs/              # File cấu hình YAML (base.yaml là config trung tâm)
configs/lambda_sweep/  # Config sweep lambda_interp cho WiSE-OD (Eq. 9)
data/                 # Dataset (không commit dữ liệu thật, xem ghi chú bên dưới)
data/scripts/          # Script tải/tiền xử lý dữ liệu
models/                # Kiến trúc mô hình (backbone, decoder, FA-VP, IAN fusion...)
losses/                # Hàm loss (Hungarian matching, cls/L1/GIoU...)
wise_od/               # Weight-Space Ensembling (nội suy trọng số, Eq. 9)
eval/                  # Script/metric đánh giá (AP, MR, robustness trên LLVIP-C)
viz/                   # Trực quan hóa kết quả, attention map, prompt tần số...
baselines/             # Cài đặt các baseline để so sánh
scripts/               # Script tiện ích (train/eval entrypoints, tools...)
tests/                 # Unit test
```

## Dữ liệu

Dữ liệu **LLVIP** và **LLVIP-C** (bản có nhiễu để đánh giá robustness) **không
được commit lên Git** vì dung lượng quá lớn. Checkpoint (`*.pth`, `*.pt`) cũng
không được commit — xem `.gitignore`.

### Chuẩn bị dữ liệu

Dữ liệu LLVIP được đặt trên **Google Drive**, mount local trên Windows dưới
dạng ổ `H:` (`H:\My Drive\Dataset\LLVIP`), cấu trúc theo repo gốc
[bupt-ai-cz/LLVIP](https://github.com/bupt-ai-cz/LLVIP):

```
H:\My Drive\Dataset\LLVIP\
  visible\train\   visible\test\    # ảnh RGB
  infrared\train\  infrared\test\   # ảnh IR, cùng tên file với visible
  Annotations\                      # XML Pascal VOC, dùng chung cho vis+ir
  annotations_coco\                 # train.json / test.json (sinh ra, không commit)
```

Vì dữ liệu nằm trên Google Drive, cùng một mount này có thể tái sử dụng khi
chuyển sang chạy trên Google Colab (mount Drive tương ứng) mà không cần upload
lại dữ liệu.

1. **Kiểm tra tính toàn vẹn dữ liệu** (đối chiếu số cặp ảnh với số liệu bài báo
   train=12,025 / test=3,463):

   ```bash
   python data/scripts/verify_pairs.py --root "H:/My Drive/Dataset/LLVIP"
   ```

2. **Convert annotation VOC XML sang COCO JSON** (ghi vào
   `H:/My Drive/Dataset/LLVIP/annotations_coco/`, không lưu trong repo):

   ```bash
   python data/scripts/convert_to_coco.py \
       --root "H:/My Drive/Dataset/LLVIP" \
       --out-dir "H:/My Drive/Dataset/LLVIP/annotations_coco"
   ```

3. **Kiểm tra lại file COCO JSON** vừa sinh bằng pycocotools:

   ```bash
   python data/scripts/check_coco.py --ann "H:/My Drive/Dataset/LLVIP/annotations_coco/train.json"
   ```

`configs/base.yaml` (`data.root`, `data.train_ann`, `data.val_ann`) đã trỏ sẵn
về các đường dẫn này.

## Trạng thái hiện tại

**Giai đoạn B - Kiểm tra dữ liệu LLVIP hoàn tất** (dữ liệu khớp 100% số liệu
bài báo, đã convert sang COCO JSON và xác thực bằng pycocotools).
