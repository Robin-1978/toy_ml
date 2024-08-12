import requests
from bs4 import BeautifulSoup
import pandas as pd

def clean_text(text):
    """移除文本中的换行符和额外的空白字符。"""
    return text.replace('\n', '').replace('\r', '').strip()

def fetch_data_from_page(page_number):
    url = f"https://caipiao.eastmoney.com/pub/Result/History/ssq?page={page_number}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            # 如果遇到404错误，返回None，None表示页码超出范围
            return None, None
        else:
            # 如果遇到其他HTTP错误，抛出异常
            raise e

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    if table is None:
        return None, None

    headers = [clean_text(header.text) for header in table.find_all('th')]
    rows = []
    for row in table.find_all('tr')[1:]:  # 跳过表头
        cols = row.find_all('td')
        cols = [clean_text(col.text) for col in cols]
        # 确保每行数据的列数与表头一致，并且行中有数据
        if any(cols):
            rows.append(cols)

    return headers, rows

def split_ball_numbers(row):
    """将第4列的号码分开成7列，并将其插入到原始数据的位置。"""
    date = row[0]  # 第一个元素是日期或其他标识信息
    numbers = row[3]  # 第四列包含号码
    # 将号码按分隔符拆分
    split_numbers = [numbers[i:i+2] for i in range(0, len(numbers), 2)]
    # 确保分列后有7个号码
    if len(split_numbers) == 7:
        return [row[0], row[1], row[2], *split_numbers, row[4], row[5], row[6], row[7], row[8], row[9]]
    else:
        return row

def main():
    all_rows = []
    page_number = 1

    while True:
        current_headers, rows = fetch_data_from_page(page_number)
        if current_headers is None and rows is None:
            print(f"第 {page_number} 页数据未找到或超出范围，程序结束。")
            break
        for row in rows:
            all_rows.append(split_ball_numbers(row))
        page_number += 1
        print(f"第 {page_number-1} 页数据抓取完成")
        break

    if all_rows:
        # 创建DataFrame，列数基于每行数据的数量
        num_columns = len(all_rows[0])  # 获取列数
        column_names = ["ID", "Date", "Detail", "Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6", "Ball_7", "Total", "Curent", "Top_Hit", "Top_Amount", "Sec_Hit", "Sec_Amount"]  # 假设的列名
        df = pd.DataFrame(all_rows, columns=column_names)
        # df = pd.DataFrame(all_rows)
        # 保存到CSV文件
        df.to_csv('ssq_data.csv', index=False, encoding='utf-8-sig')
        print("所有数据已成功保存到 'ssq_data.csv'")
    else:
        print("未抓取到任何数据。")

if __name__ == "__main__":
    main()
