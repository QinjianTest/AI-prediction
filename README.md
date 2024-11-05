# 黄金价格预测程序
现公司应用 2024

AI使用示例，使用线性回归模型预测黄金价格

## 简介

通过历史黄金价格数据训练一个线性回归模型，从而预测未来的黄金价格，输出过去5天的实际价格和未来10天的预测价格。

## 使用说明
使用以下命令安装依赖项：
```bash
pip install -r requirements.txt
```
     
准备一个名为gold_prices.csv的CSV文件，包含日期和价格列，下载地址为
```bash
https://www.investing.com/commodities/gold-historical-data
```
运行程序：
```bash
python getPrice.py
```

