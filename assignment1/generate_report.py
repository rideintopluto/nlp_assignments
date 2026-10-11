"""Generate a Chinese PDF report from real training results."""

from __future__ import annotations

import argparse
import os
import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据训练结果生成实验报告 PDF")
    parser.add_argument("--results", type=Path, default=Path("artifacts/results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/实验报告.pdf"))
    parser.add_argument("--student-id", required=True, help="学号")
    parser.add_argument("--student-name", required=True, help="姓名")
    parser.add_argument("--course", default="自然语言处理", help="课程名称")
    parser.add_argument("--teacher", default="", help="任课教师，可省略")
    parser.add_argument("--font", type=Path, help="可选的中文 TTF/TTC 字体路径")
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def seconds_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    return f"{seconds / 60:.1f} 分钟"


def analysis_text(results: dict) -> str:
    train_accuracy = float(results["best_train_accuracy"])
    val_accuracy = float(results["validation_accuracy"])
    gap = train_accuracy - val_accuracy
    cat_accuracy = float(results["per_class_accuracy"]["cat"])
    dog_accuracy = float(results["per_class_accuracy"]["dog"])

    if train_accuracy < 0.65 and val_accuracy < 0.65:
        fit_comment = "训练集和验证集准确率都相对有限，模型可能仍有欠拟合现象。可尝试适当增加训练轮数、调整学习率或扩展网络容量。"
    elif gap > 0.10:
        fit_comment = "训练准确率明显高于验证准确率，说明模型存在一定过拟合。当前实验已使用数据增强、Dropout、权重衰减和早停来缓解该问题。"
    else:
        fit_comment = "训练集与验证集准确率差距处于较合理范围，训练过程没有表现出明显的过拟合。"

    difference = abs(cat_accuracy - dog_accuracy)
    if difference < 0.05:
        class_comment = "猫、狗两个类别的准确率较为接近，模型对两类样本的识别较均衡。"
    else:
        weaker = "猫" if cat_accuracy < dog_accuracy else "狗"
        class_comment = f"两个类别的准确率存在一定差异，其中{weaker}类别较低。可能原因包括类内姿态、背景和光照差异，或部分图片特征相近。"
    return fit_comment + class_comment


def draw_page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, str(document.page))
    canvas.restoreState()


def register_chinese_fonts(custom_font: Path | None) -> tuple[str, str]:
    """Register an embedded CJK font, preferring fonts available on Windows."""
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    regular_candidates = [
        custom_font,
        windows_fonts / "msyh.ttc",
        windows_fonts / "simsun.ttc",
        windows_fonts / "deng.ttf",
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    bold_candidates = [
        custom_font,
        windows_fonts / "msyhbd.ttc",
        windows_fonts / "simhei.ttf",
        windows_fonts / "dengb.ttf",
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ]

    try:
        import japanize_matplotlib

        bundled_font = Path(japanize_matplotlib.__file__).parent / "fonts" / "ipaexg.ttf"
        regular_candidates.append(bundled_font)
        bold_candidates.append(bundled_font)
    except ImportError:
        pass

    regular_path = next((path for path in regular_candidates if path and path.is_file()), None)
    bold_path = next((path for path in bold_candidates if path and path.is_file()), regular_path)
    if regular_path is None:
        raise FileNotFoundError(
            "没有找到可嵌入 PDF 的中文字体。Windows 通常自带微软雅黑；"
            "也可以使用 --font 指定一个支持中文的 TTF 或 TTC 字体。"
        )

    pdfmetrics.registerFont(TTFont("Chinese", str(regular_path), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ChineseBold", str(bold_path), subfontIndex=0))
    return "Chinese", "ChineseBold"


def build_report(args: argparse.Namespace, results: dict) -> None:
    regular_font, bold_font = register_chinese_fonts(args.font)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        rightMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.8 * cm,
        title="基于卷积神经网络的猫狗图像分类实验报告",
        author=args.student_name,
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChineseTitle",
        parent=base["Title"],
        fontName=bold_font,
        fontSize=22,
        leading=31,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=24,
    )
    subtitle_style = ParagraphStyle(
        "ChineseSubtitle",
        parent=base["Normal"],
        fontName=regular_font,
        fontSize=12,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#444444"),
    )
    heading_style = ParagraphStyle(
        "ChineseHeading",
        parent=base["Heading1"],
        fontName=bold_font,
        fontSize=15,
        leading=22,
        textColor=colors.black,
        spaceBefore=13,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ChineseBody",
        parent=base["BodyText"],
        fontName=regular_font,
        fontSize=10.5,
        leading=18,
        firstLineIndent=2 * 10.5,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceAfter=7,
    )
    caption_style = ParagraphStyle(
        "ChineseCaption",
        parent=base["Normal"],
        fontName=regular_font,
        fontSize=9.5,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=8,
    )
    table_text_style = ParagraphStyle(
        "ChineseTable",
        parent=base["Normal"],
        fontName=regular_font,
        fontSize=9.2,
        leading=14,
    )
    table_header_style = ParagraphStyle(
        "ChineseTableHeader",
        parent=table_text_style,
        fontName=bold_font,
        textColor=colors.white,
        alignment=TA_CENTER,
    )

    def p(text: str, style=body_style) -> Paragraph:
        return Paragraph(text, style)

    def styled_table(rows, widths):
        formatted = [
            [p(str(cell), table_header_style if row_index == 0 else table_text_style) for cell in row]
            for row_index, row in enumerate(rows)
        ]
        table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="CENTER")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244A73")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        for index in range(2, len(rows), 2):
            table.setStyle(TableStyle([("BACKGROUND", (0, index), (-1, index), colors.HexColor("#F3F6FA"))]))
        return table

    dataset = results["dataset"]
    hp = results["hyperparameters"]
    environment = results["environment"]
    matrix = results["confusion_matrix"]
    story = [
        Spacer(1, 1.5 * cm),
        p("基于卷积神经网络的猫狗图像分类", title_style),
        p("课程实验报告", subtitle_style),
        Spacer(1, 2.3 * cm),
        styled_table(
            [
                ["项目", "信息"],
                ["课程名称", args.course],
                ["学生姓名", args.student_name],
                ["学号", args.student_id],
                ["任课教师", args.teacher or "未填写"],
                ["完成日期", date.today().isoformat()],
            ],
            [4.2 * cm, 9.0 * cm],
        ),
        PageBreak(),
        p("1 实验目的", heading_style),
        p(
            "本实验使用 PyTorch 构建卷积神经网络，对给定猫狗图片进行二分类。实验目标是掌握图像数据的读取与预处理、卷积神经网络的设计、训练与验证流程，以及准确率、分类别准确率和混淆矩阵等评价方法。",
        ),
        p("2 数据集与预处理", heading_style),
        p(
            f"数据集共包含 {dataset['train_total']} 张训练图片和 {dataset['val_total']} 张验证图片。训练集包括 "
            f"{dataset['train_counts']['cat']} 张猫图片和 {dataset['train_counts']['dog']} 张狗图片；验证集包括 "
            f"{dataset['val_counts']['cat']} 张猫图片和 {dataset['val_counts']['dog']} 张狗图片。",
        ),
        p(
            f"所有图片首先转换为 RGB 格式并缩放至 {hp['image_size']}×{hp['image_size']}。训练阶段随机进行水平翻转、±10 度旋转以及轻微亮度和对比度扰动，以增加样本多样性；验证阶段只进行尺寸调整。最后将像素归一化到近似 [-1, 1] 的范围。",
        ),
        p("3 网络结构", heading_style),
        p(
            "本实验从零训练一个四层卷积神经网络。每个卷积模块由 3×3 卷积、批归一化、ReLU 激活和 2×2 最大池化组成。卷积通道依次为 32、64、128 和 256。特征提取后使用自适应平均池化，将空间特征压缩为固定长度向量，再通过含 Dropout 的全连接层输出猫、狗两个类别的得分。",
        ),
        styled_table(
            [
                ["层次", "主要操作", "输出通道"],
                ["卷积模块 1", "3×3 卷积 + BN + ReLU + 最大池化", "32"],
                ["卷积模块 2", "3×3 卷积 + BN + ReLU + 最大池化", "64"],
                ["卷积模块 3", "3×3 卷积 + BN + ReLU + 最大池化", "128"],
                ["卷积模块 4", "3×3 卷积 + BN + ReLU + 最大池化", "256"],
                ["分类器", "自适应平均池化 + 全连接 + Dropout", "2"],
            ],
            [3.1 * cm, 8.0 * cm, 2.1 * cm],
        ),
        Spacer(1, 0.3 * cm),
        p(f"模型可训练参数总量为 {results['model_summary']['trainable_parameters']:,}，未使用任何预训练权重。"),
        p("4 实验设置", heading_style),
        styled_table(
            [
                ["配置项", "取值"],
                ["深度学习框架", f"PyTorch {environment['pytorch_version']}"],
                ["训练设备", f"{environment['device']} / {environment['device_name']}"],
                ["批大小", hp["batch_size"]],
                ["优化器", hp["optimizer"]],
                ["初始学习率", hp["learning_rate"]],
                ["损失函数", hp["loss_function"]],
                ["权重衰减", hp["weight_decay"]],
                ["Dropout", hp["dropout"]],
                ["随机种子", hp["seed"]],
                ["实际训练轮数", hp["completed_epochs"]],
            ],
            [5.0 * cm, 8.2 * cm],
        ),
        p("5 实验结果", heading_style),
        p(
            f"模型在第 {results['best_epoch']} 轮取得最佳验证结果，训练耗时为 {seconds_text(results['training_seconds'])}。"
            f"验证集总体准确率为 {percent(results['validation_accuracy'])}，猫类别准确率为 "
            f"{percent(results['per_class_accuracy']['cat'])}，狗类别准确率为 {percent(results['per_class_accuracy']['dog'])}。",
        ),
        styled_table(
            [
                ["指标", "结果"],
                ["最佳轮次", results["best_epoch"]],
                ["最佳轮次训练损失", f"{results['best_train_loss']:.4f}"],
                ["最佳轮次训练准确率", percent(results["best_train_accuracy"])],
                ["验证损失", f"{results['best_val_loss']:.4f}"],
                ["验证集总体准确率", percent(results["validation_accuracy"])],
                ["猫类别准确率", percent(results["per_class_accuracy"]["cat"])],
                ["狗类别准确率", percent(results["per_class_accuracy"]["dog"])],
            ],
            [6.8 * cm, 6.4 * cm],
        ),
    ]

    results_dir = args.results.parent
    curve_path = results_dir / "training_curves.png"
    matrix_path = results_dir / "confusion_matrix.png"
    if curve_path.is_file():
        story.extend(
            [
                Spacer(1, 0.4 * cm),
                Image(str(curve_path), width=15.0 * cm, height=5.73 * cm),
                p("图 1 训练集与验证集的损失和准确率曲线", caption_style),
            ]
        )
    if matrix_path.is_file():
        story.extend(
            [
                Image(str(matrix_path), width=8.8 * cm, height=7.62 * cm),
                p("图 2 验证集混淆矩阵", caption_style),
            ]
        )

    story.extend(
        [
            p("6 结果分析与思考", heading_style),
            p(analysis_text(results)),
            p(
                f"混淆矩阵中，真实猫图片被正确识别为猫的数量为 {matrix[0][0]}，误判为狗的数量为 {matrix[0][1]}；"
                f"真实狗图片被误判为猫的数量为 {matrix[1][0]}，正确识别为狗的数量为 {matrix[1][1]}。"
                "错误样本可能具有复杂背景、主体较小、遮挡或姿态特殊等特点。后续可以查看误分类图片，并针对性调整数据增强和网络结构。",
            ),
            p(
                "本实验让我熟悉了从数据加载、网络构建、损失计算、反向传播到验证评估的完整深度学习流程。"
                "卷积层能够保留并提取图像的局部空间特征，因此比直接展平像素的全连接网络更适合本任务。"
                "实验也表明，除模型结构外，数据预处理、随机种子、学习率、正则化和训练轮数都会影响最终结果。",
            ),
            p("7 实验结论", heading_style),
            p(
                f"本实验使用 PyTorch 独立实现并训练了 CNN 猫狗二分类模型，在 {dataset['val_total']} 张验证图片上取得 "
                f"{percent(results['validation_accuracy'])} 的总体准确率。实验完成了数据预处理、CNN 建模、训练验证、"
                "分类别评价与结果可视化，达到了掌握深度学习图像分类基本流程的目的。",
            ),
        ]
    )

    document.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)


def main() -> None:
    args = parse_args()
    if not args.results.is_file():
        raise FileNotFoundError(f"找不到训练结果：{args.results}。请先运行 train.py 完成训练。")
    with args.results.open("r", encoding="utf-8") as stream:
        results = json.load(stream)
    build_report(args, results)
    print(f"实验报告已生成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
