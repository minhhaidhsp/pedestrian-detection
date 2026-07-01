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

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **Lưu ý (Windows):** `pycocotools` và `torch` cần wheel tương thích với phiên
> bản Python đang dùng. Nếu pip không tìm được wheel dựng sẵn, `pycocotools` sẽ
> cố build từ source và cần Visual C++ Build Tools. Nếu build lỗi, có thể thử
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
được commit lên Git** vì dung lượng quá lớn. Hướng dẫn tải và tổ chức dữ liệu
sẽ được bổ sung ở **Giai đoạn B** (chưa thực hiện ở bước khởi tạo repo này).

Checkpoint (`*.pth`, `*.pt`) cũng không được commit — xem `.gitignore`.

## Trạng thái hiện tại

**Giai đoạn A - Repo skeleton hoàn tất.**
