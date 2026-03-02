import sys
import os
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel
from PyQt5.QtCore import QObject, pyqtSignal
import sqlite3


class SqliteDb(QObject):
    """
        自定义类，通过pyqt5的数据库接口实现数据库以下操作
        """

    # 定义信号，用于异步操作完成时发送结果
    operation_finished = pyqtSignal(bool, str)  # (success, message)
    data_retrieved = pyqtSignal(list)  # 发送查询结果

    def __init__(self, db_file="./FaceDatabase.db"):
        """
                初始化数据库连接
                :param db_file: 数据库文件路径
                """
        super().__init__()
        self.db_file = db_file
        self.db = None
        self.init_database()

    def init_database(self):
        """
                初始化数据库连接
                """
        try:
            # 创建数据库连接
            self.db = QSqlDatabase.addDatabase("QSQLITE")
            self.db.setDatabaseName(self.db_file)

            if not self.db.open():
                raise Exception(f"无法打开数据库连接: {self.db.lastError().text()}")

            print(f"已连接到数据库: {self.db_file}")
            self.create_table()
        except Exception as e:
            print(f"初始化数据库失败: {str(e)}")

    def create_table(self):
        """
                重新创建一个表并添加id(主键)，姓名，时间，性别，学号，录入时间，照片字段
                """
        try:
            query = QSqlQuery(self.db)

            # 删除旧表（如果存在）
            drop_sql = "DROP TABLE IF EXISTS facedatabase"
            if not query.exec_(drop_sql):
                print(f"删除旧表失败: {query.lastError().text()}")

            # 创建新表
            create_sql = """
                CREATE TABLE facedatabase (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    姓名 TEXT NOT NULL,
                    时间 TEXT,
                    性别 TEXT,
                    学号 INTEGER,
                    录入时间 TEXT,
                    照片 BLOB
                );
            """

            if not query.exec_(create_sql):
                raise Exception(f"创建表失败: {query.lastError().text()}")

            print("表创建成功")
            return True

        except Exception as e:
            print(f"创建表时发生错误: {str(e)}")
            return False

    def search_table(self, name_keyword):
        """
                能通过姓名匹配出对应的姓名，时间，性别，学号，录入时间，照片字段内容
                :param name_keyword: 姓名关键字，支持模糊匹配
                :return: 匹配的记录列表，每条记录为元组 (姓名, 时间, 性别, 学号, 录入时间, 照片)
                """
        try:
            query = QSqlQuery(self.db)

            # 使用LIKE进行模糊匹配
            sql = """
                SELECT 姓名, 时间, 性别, 学号, 录入时间, 照片 
                FROM facedatabase 
                WHERE 姓名 LIKE ?
            """

            query.prepare(sql)
            query.addBindValue(f"%{name_keyword}%")

            if not query.exec_():
                raise Exception(f"查询失败: {query.lastError().text()}")

            # 收集查询结果
            results = []
            while query.next():
                record = []
                for i in range(query.record().count()):
                    record.append(query.value(i))
                results.append(tuple(record))

            print(f"查询到 {len(results)} 条记录")
            return results

        except Exception as e:
            print(f"查询时发生错误: {str(e)}")
            return []

    def add_table(self, name, time, gender, student_id, entry_time, photo):
        """
                能将姓名，时间，性别，学号，录入时间，照片加入数据库
                :param name: 姓名
                :param time: 时间
                :param gender: 性别
                :param student_id: 学号
                :param entry_time: 录入时间
                :param photo: 照片（BLOB数据）
                :return: 成功返回True，失败返回False
                """
        try:
            query = QSqlQuery(self.db)

            sql = """
                INSERT INTO facedatabase (姓名, 时间, 性别, 学号, 录入时间, 照片)
                VALUES (?, ?, ?, ?, ?, ?)
            """

            query.prepare(sql)
            query.addBindValue(name)
            query.addBindValue(time)
            query.addBindValue(gender)
            query.addBindValue(student_id)
            query.addBindValue(entry_time)
            query.addBindValue(photo)

            if not query.exec_():
                raise Exception(f"插入数据失败: {query.lastError().text()}")

            print("数据插入成功")
            return True

        except Exception as e:
            print(f"插入数据时发生错误: {str(e)}")
            return False

    def get_all_records(self):
        """
                获取所有记录，用于测试和显示
                :return: 所有记录列表
                """
        try:
            query = QSqlQuery(self.db)

            sql = "SELECT * FROM facedatabase"
            if not query.exec_(sql):
                raise Exception(f"查询所有记录失败: {query.lastError().text()}")

            results = []
            while query.next():
                record = []
                for i in range(query.record().count()):
                    record.append(query.value(i))
                results.append(tuple(record))

            return results

        except Exception as e:
            print(f"获取所有记录时发生错误: {str(e)}")
            return []


# 示例使用代码
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # 创建数据库实例
    db_manager = SqliteDb()

    # 测试创建表
    print("=== 测试创建表 ===")
    db_manager.create_table()

    # 测试添加数据
    print("\n=== 测试添加数据 ===")
    # 添加一些示例数据
    sample_photo_data = b"This is a sample photo data"  # 实际应用中应为真实的图片字节数据
    success = db_manager.add_table(
        name="张三",
        time="2023-10-01 10:30:00",
        gender="男",
        student_id=20230001,
        entry_time="2023-10-01 10:30:00",
        photo=sample_photo_data
    )
    print(f"添加数据结果: {success}")

    success = db_manager.add_table(
        name="李四",
        time="2023-10-01 10:35:00",
        gender="女",
        student_id=20230002,
        entry_time="2023-10-01 10:35:00",
        photo=sample_photo_data
    )
    print(f"添加数据结果: {success}")

    success = db_manager.add_table(
        name="王五",
        time="2023-10-01 10:40:00",
        gender="男",
        student_id=20230003,
        entry_time="2023-10-01 10:40:00",
        photo=sample_photo_data
    )
    print(f"添加数据结果: {success}")

    # 测试查询数据
    print("\n=== 测试查询数据 ===")
    results = db_manager.search_table("张")
    print(f"查询结果: {results}")

    results = db_manager.search_table("李")
    print(f"查询结果: {results}")

    results = db_manager.search_table("王")
    print(f"查询结果: {results}")

    # 测试模糊查询
    print("\n=== 测试模糊查询 ===")
    results = db_manager.search_table("张三")  # 完全匹配
    print(f"完全匹配查询结果: {results}")

    results = db_manager.search_table("三")  # 模糊匹配
    print(f"模糊匹配查询结果: {results}")

    print("\n=== 获取所有记录 ===")
    all_records = db_manager.get_all_records()
    for record in all_records:
        print(record)

    print("\n数据库操作测试完成")