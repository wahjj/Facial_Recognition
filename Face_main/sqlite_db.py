from PyQt5.QtSql import QSqlQuery, QSqlDatabase
from PyQt5.QtCore import QByteArray


class MySqlite:
    def __init__(self):
        try:
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            self.database.setDatabaseName("student.db")
            self.database.open()
            self.query = QSqlQuery()
        except Exception as e:
            print(f"无法建立连接，error:{e}")

    def operation_sql(self, sql, params=None):
        try:
            if params:
                if not self.query.prepare(sql):
                    print(f"SQL准备失败: {self.query.lastError().text()}")
                    return False
                for param in params:
                    # 处理 BLOB 数据
                    if isinstance(param, bytes):
                        self.query.addBindValue(QByteArray(param))
                    else:
                        self.query.addBindValue(param)
                if not self.query.exec_():
                    print(f"SQL执行失败: {self.query.lastError().text()}")
                    return False
            else:
                if not self.query.exec_(sql):
                    print(f"SQL执行失败: {self.query.lastError().text()}")
                    return False

            if sql.strip().upper().startswith("SELECT"):
                results = []
                while self.query.next():
                    row = []
                    for i in range(self.query.record().count()):
                        value = self.query.value(i)
                        # 将 QByteArray 转回 bytes (如果需要)，或者直接保留
                        if isinstance(value, QByteArray):
                            value = bytes(value)
                        row.append(value)
                    results.append(row)
                return results
            else:
                return True

        except Exception as e:
            print(f"操作SQL异常: {str(e)}")
            return False


if __name__ == '__main__':
    db = MySqlite()

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
    print(ret)