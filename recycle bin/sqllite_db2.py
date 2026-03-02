import sys
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlError
from PyQt5.QtCore import QDate, QTime, QDateTime, QFile, QIODevice
from PyQt5.QtGui import QPixmap


class StudentManager:
    def __init__(self, db_name="student_db.sqlite"):
        """
        初始化数据库连接
        :param db_name: 数据库文件名
        """
        # 保存数据库文件名
        self.db_name = db_name

        # 添加SQLite数据库驱动
        self.db = QSqlDatabase.addDatabase("QSQLITE")

        # 设置数据库文件名
        self.db.setDatabaseName(db_name)

        # 尝试打开数据库连接
        if not self.db.open():
            # 如果连接失败，打印错误信息并抛出异常
            print(f"数据库打开失败: {self.db.lastError().text()}")
            raise Exception("无法连接数据库")
        else:
            # 连接成功提示
            print(f"数据库连接成功: {db_name}")

    def create_table(self):
        """
        重新创建一个表。如果表已存在则先删除。
        字段: id(主键), 姓名, 时间, 性别, 学号, 录入时间, 照片(BLOB)
        """
        # 创建SQL查询对象
        query = QSqlQuery()

        # 1. 如果表存在，先删除 (实现"重新创建")
        drop_query = "DROP TABLE IF EXISTS students"
        if not query.exec(drop_query):
            print(f"删除旧表失败: {query.lastError().text()}")
            return False

        # 2. 创建新表
        # 注意：照片字段使用 BLOB 类型存储二进制数据
        create_sql = """
        CREATE TABLE students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
            name TEXT NOT NULL,                    -- 姓名（必填）
            time TEXT,                             -- 时间
            gender TEXT,                           -- 性别
            student_id TEXT,                       -- 学号
            entry_time TEXT,                       -- 录入时间
            photo BLOB                             -- 照片（二进制大对象）
        )
        """
        if not query.exec(create_sql):
            print(f"创建表失败: {query.lastError().text()}")
            return False

        print("表 'students' 创建成功 (id, 姓名, 时间, 性别, 学号, 录入时间, 照片)")
        return True

    def add_table(
        self, name, time_str, gender, student_id, entry_time_str, photo_path=None
    ):
        """
        添加数据到数据库
        :param name: 姓名
        :param time_str: 时间 (字符串)
        :param gender: 性别
        :param student_id: 学号
        :param entry_time_str: 录入时间 (字符串)
        :param photo_path: 照片文件路径 (可选)
        :return: 成功返回True，否则False
        """
        # 创建查询对象
        query = QSqlQuery()

        # 处理照片数据：读取文件转为二进制
        photo_blob = None
        # 检查是否有照片路径且文件存在
        if photo_path and QFile.exists(photo_path):
            # 创建文件对象
            file = QFile(photo_path)
            # 以只读模式打开文件
            if file.open(QIODevice.ReadOnly):
                # 读取全部文件内容
                photo_blob = file.readAll()
                # 关闭文件
                file.close()
            else:
                print(f"警告：无法读取照片文件 {photo_path}")

        # 构建插入语句
        # 使用占位符 :name, :time 等防止 SQL 注入
        sql = """
        INSERT INTO students (name, time, gender, student_id, entry_time, photo)
        VALUES (:name, :time, :gender, :sid, :entry_time, :photo)
        """

        # 准备SQL语句（预编译）
        query.prepare(sql)
        # 绑定参数值
        query.bindValue(":name", name)
        query.bindValue(":time", time_str)
        query.bindValue(":gender", gender)
        query.bindValue(":sid", student_id)
        query.bindValue(":entry_time", entry_time_str)

        # 根据照片数据是否存在绑定不同的值
        if photo_blob is not None:
            query.bindValue(":photo", photo_blob)
        else:
            query.bindValue(":photo", None)

        # 执行插入操作
        if not query.exec():
            print(f"添加数据失败: {query.lastError().text()}")
            return False

        print(f"成功添加记录: {name}")
        return True

    def search_table(self, name):
        """
        通过姓名匹配查询
        :param name: 要搜索的姓名 (支持模糊匹配，这里使用精确匹配，如需模糊可改为 LIKE)
        :return: 返回包含字典列表的结果，每个字典代表一行数据
        """
        # 创建查询对象
        query = QSqlQuery()

        # 使用精确匹配，如果需要模糊匹配可使用: f"SELECT * FROM students WHERE name LIKE '%{name}%'"
        # 为了安全起见，依然推荐使用 bindValue 即使是在 LIKE 中，但这里演示精确匹配
        sql = "SELECT id, name, time, gender, student_id, entry_time, photo FROM students WHERE name = :name"
        # 准备SQL语句
        query.prepare(sql)
        # 绑定姓名参数
        query.bindValue(":name", name)

        # 结果列表
        results = []

        # 执行查询
        if not query.exec():
            print(f"查询失败: {query.lastError().text()}")
            return results

        # 获取各字段在结果集中的索引位置
        idx_id = query.record().indexOf("id")
        idx_name = query.record().indexOf("name")
        idx_time = query.record().indexOf("time")
        idx_gender = query.record().indexOf("gender")
        idx_sid = query.record().indexOf("student_id")
        idx_entry = query.record().indexOf("entry_time")
        idx_photo = query.record().indexOf("photo")

        # 遍历查询结果
        while query.next():
            # 创建字典存储当前行的数据
            row_data = {
                "id": query.value(idx_id),  # ID
                "姓名": query.value(idx_name),  # 姓名
                "时间": query.value(idx_time),  # 时间
                "性别": query.value(idx_gender),  # 性别
                "学号": query.value(idx_sid),  # 学号
                "录入时间": query.value(idx_entry),  # 录入时间
                "照片数据": query.value(idx_photo),  # 照片数据（QByteArray对象）
            }
            # 将当前行数据添加到结果列表
            results.append(row_data)

        # 输出查询结果统计信息
        if not results:
            print(f"未找到姓名为 '{name}' 的记录")
        else:
            print(f"找到 {len(results)} 条记录")

        return results

    def close(self):
        """关闭数据库连接"""
        # 关闭数据库连接
        self.db.close()
        print("数据库连接已关闭")


# --- 测试演示部分 ---
if __name__ == "__main__":
    # PyQt5 的 SQL 模块需要一个 QApplication 实例才能正常工作（即使在命令行脚本中）
    app = sys.modules.get("QApplication")
    from PyQt5.QtWidgets import QApplication

    if not app:
        app = QApplication(sys.argv)

    # 1. 实例化管理器
    manager = StudentManager("student_db.sqlite")

    # 2. 创建/重置表
    print("\n--- 执行 create_table ---")
    manager.create_table()

    # 3. 添加数据 (add_table)
    print("\n--- 执行 add_table ---")
    # 导入datetime模块获取当前时间
    from datetime import datetime

    # 格式化当前时间为字符串
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 注意：如果没有真实的图片文件，photo_path 可以传 None，数据库中该字段将为 NULL
    # 为了演示，我们假设有一个 test.jpg，如果没有，代码也会正常运行只是照片为空
    success = manager.add_table(
        name="张三",
        time_str="2026-02-28 09:00",
        gender="男",
        student_id="2023001",
        entry_time_str=now_str,
        photo_path=None,
    )

    # 添加第二条记录
    manager.add_table(
        name="李四",
        time_str="2026-02-28 09:05",
        gender="女",
        student_id="2023002",
        entry_time_str=now_str,
        photo_path=None,
    )

    # 4. 搜索数据 (search_table)
    print("\n--- 执行 search_table (搜索 '张三') ---")
    results = manager.search_table("张三")

    # 遍历并打印查询结果
    for row in results:
        print(f"ID: {row['id']}, 姓名: {row['姓名']}, 学号: {row['学号']}, 性别: {row['性别']}")
        # 检查是否有照片数据
        if row["照片数据"]:
            print("  -> 照片数据已加载 (BLOB)")
        else:
            print("  -> 无照片")

    # 清理：关闭数据库连接
    manager.close()
