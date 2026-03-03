# 导入PyQt5中用于数据库操作的模块
from PyQt5.QtSql import QSqlQuery, QSqlDatabase
# 导入PyQt5中用于处理二进制数据的模块
from PyQt5.QtCore import QByteArray


class MySqlite:
    """
    自定义SQLite数据库操作类，封装了数据库连接、SQL执行、查询等功能。
    """

    def __init__(self):
        """
        初始化数据库连接。
        - 添加SQLite数据库驱动。
        - 设置数据库名称为 "student.db"。
        - 打开数据库连接。
        - 创建一个SQL查询对象。
        """
        try:
            # 添加SQLite数据库驱动
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            # 设置数据库文件名
            self.database.setDatabaseName("student.db")
            # 打开数据库连接
            self.database.open()
            # 创建SQL查询对象
            self.query = QSqlQuery()
        except Exception as e:
            # 捕获异常并打印错误信息
            print(f"无法建立连接，error:{e}")

    def operation_sql(self, sql, params=None):
        """
        执行SQL语句，支持带参数的查询和非参数化查询。

        参数:
        - sql: 要执行的SQL语句。
        - params: 可选参数列表，用于绑定到SQL语句中的占位符。

        返回值:
        - 成功时返回True。
        - 失败时返回错误信息。
        """
        if params:
            # 准备带参数的SQL语句
            result = self.query.prepare(sql)
            if not result:
                # 如果准备失败，返回错误信息
                return self.query.lastError().text()
            for param in params:
                # 对于BLOB数据，需要转换为QByteArray
                if isinstance(param, bytes):
                    qbyte_array = QByteArray(param)
                    self.query.addBindValue(qbyte_array)
                else:
                    self.query.addBindValue(param)
            # 执行SQL语句
            result = self.query.exec_()
        else:
            # 直接执行SQL语句（无参数）
            result = self.query.exec_(sql)

        if result:
            # 执行成功返回True
            return True
        else:
            # 执行失败返回错误信息
            return self.query.lastError().text()

    def select_name(self, name):
        """
        根据姓名查询学生信息。

        参数:
        - name: 学生姓名。

        功能:
        - 使用参数化查询防止SQL注入。
        - 查询结果逐行输出字段值。
        """
        # 使用参数化查询防止SQL注入
        result = self.operation_sql("select * from student_info WHERE 姓名 = ?", [name])
        if result:
            # 获取查询结果的记录结构
            self.result = self.query.record()
            # 遍历查询结果
            while self.query.next():
                for i in range(self.result.count()):
                    # 输出每个字段的值
                    print(str(self.query.value(i)))
        else:
            # 查询失败抛出异常
            raise Exception(result)

    def export_photo(self, name):
        """
        根据姓名导出学生的照片。

        参数:
        - name: 学生姓名。

        功能:
        - 查询指定学生的照片字段（BLOB类型）。
        - 将照片保存为本地文件 "exported_photo.jpg"。
        """
        # 使用参数化查询防止SQL注入
        self.operation_sql("SELECT 照片 FROM student_info WHERE 姓名 = ?", [name])
        if self.query.next():
            # 获取照片字段的数据
            blob_data = self.query.value(0)
            if blob_data:
                # 如果blob_data是QByteArray，则需要转换为bytes
                if hasattr(blob_data, 'data'):
                    blob_data = blob_data.data()
                # 将照片写入文件
                with open("exported_photo.jpg", "wb") as f:
                    f.write(blob_data)
                print("图片已导出为 exported_photo.jpg")
            else:
                print("照片字段为空")


if __name__ == '__main__':
    # 实例化数据库操作类
    db = MySqlite()

    # 创建表（如果不存在）
    ret = db.operation_sql("""
        create table IF NOT EXISTS student_info(
            ID integer primary key AUTOINCREMENT,
            姓名 text,
            年龄 int,
            性别 text,
            学号 text,
            录入时间 text,
            照片 BLOB
        )
    """)
    print(ret)  # 输出创建表的结果

    # 读取一张图片作为测试数据
    with open(r'/dataset/images/test/Ziwang_Xu_0001.jpg', 'rb') as f:
        img = f.read()

    # 插入一条学生信息记录
    ret = db.operation_sql(
        """
        INSERT INTO student_info (ID, 姓名, 年龄, 性别, 学号, 录入时间, 照片)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [None, '张三', 23, '男', '244345544', '2026-02-28-11:15', img]
    )
    print(ret)  # 输出插入结果

    # 查询学生信息
    ret = db.select_name("张三")
    print(ret)  # 输出查询结果

    # 导出学生照片
    ret = db.export_photo("张三")

    # 其他可选操作（被注释掉）：
    # 删除记录
    # ret = db.operation_sql(f"delete from student_info where ID = 4")
    # print(ret)
