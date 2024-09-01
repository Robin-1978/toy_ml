import requests
from bs4 import BeautifulSoup
import DataModel
import re


def clean_text(text):
    """移除文本中的换行符和额外的空白字符。"""
    return text.replace("\n", "").replace("\r", "").strip()


def fetch_data_from_page(page_number):
    url = f"https://caipiao.eastmoney.com/pub/Result/History/fc3d?page={page_number}"
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

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return None, None

    headers = [clean_text(header.text) for header in table.find_all("th")]
    rows = []
    for row in table.find_all("tr")[1:]:  # 跳过表头
        cols = row.find_all("td")
        cols = [clean_text(col.text) for col in cols]
        # 确保每行数据的列数与表头一致，并且行中有数据
        if any(cols):
            rows.append(cols)

    return headers, rows


def split_ball_numbers(row):
    date_week = re.search(r"(\d{4}-\d{2}-\d{2})\((星期\w+)\)", row[1])
    date_week = date_week.group(1, 2)

    # 由于福彩3D只有三个号码，所以这里拆分为3个数字
    numbers = row[3]  # 第四列包含号码
    split_numbers = [numbers[i : i + 1] for i in range(0, len(numbers), 1)]
    # 确保分列后有3个号码
    if len(split_numbers) == 3:
        return [
            row[0],
            *date_week,
            row[2],
            *split_numbers,
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
        ]
    else:
        return row


def main():
    all_rows = []
    page_number = 1
    session = DataModel.ConnectDB("data/fc3d.db")
    while True:
        current_headers, rows = fetch_data_from_page(page_number)
        if current_headers is None and rows is None:
            print(f"第 {page_number} 页数据未找到或超出范围，程序结束。")
            break
        for row in rows:
            all_rows.append(split_ball_numbers(row))
        page_number += 1
        print(f"第 {page_number-1} 页数据抓取完成")
        # add rows to database when the key ID doesn't exist
        isExist = False
        for row in rows:
            row = split_ball_numbers(row)
            if session.query(DataModel.Fc3d).filter_by(ID=row[0]).first() is None:
                fc3d = DataModel.Fc3d(
                    **{
                        "ID": row[0],
                        "Date": row[1],
                        "Week": row[2],
                        "Detail": row[3],
                        "Ball_1": row[4],
                        "Ball_2": row[5],
                        "Ball_3": row[6],
                        "Total": row[7],
                        "Curent": row[8],
                        "TopHit": row[9],
                        "TopAmount": row[10],
                        "SecHit": row[11],
                        "SecAmount": row[12],
                    }
                )
                session.add(fc3d)
                print(f"已添加 {row[0]}")
            else:
                # print(f"已存在 {row[0]}")
                isExist = True
                # 退出下载
        if isExist:
            break
    session.commit()
    session.close()
    print("数据抓取完成")


if __name__ == "__main__":
    main()
