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
        try:
            if params:
                # 准备带参数的SQL语句
                if not self.query.prepare(sql):
                    return self.query.lastError().text()
                for param in params:
                    # 处理 BLOB 数据
                    if isinstance(param, bytes):
                        self.query.addBindValue(QByteArray(param))
                    else:
                        self.query.addBindValue(param)
                # 执行
                if not self.query.exec_():
                    return self.query.lastError().text()
            else:
                # 无参数直接执行
                if not self.query.exec_(sql):
                    return self.query.lastError().text()
            # 判断是否为查询语句 (SELECT)
            if sql.strip().upper().startswith("SELECT"):
                results = []
                # 遍历结果集并收集数据
                while self.query.next():
                    row = []
                    # 获取当前行的所有列值
                    for i in range(self.query.record().count()):
                        value = self.query.value(i)
                        # 将 QByteArray 转回 bytes (如果需要)，或者直接保留
                        if isinstance(value, QByteArray):
                            value = bytes(value)
                        row.append(value)
                    results.append(row)
                return results  # 返回数据列表
            else:
                return True   # 非查询语句成功则返回 True

        except Exception as e:
            return str(e)

    def select_name(self, name):
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
