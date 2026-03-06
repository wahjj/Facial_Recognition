# db.py
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
                    return False
                for param in params:
                    if isinstance(param, bytes):
                        self.query.addBindValue(QByteArray(param))
                    else:
                        self.query.addBindValue(param)
                if not self.query.exec_():
                    return False
            else:
                if not self.query.exec_(sql):
                    return False

            if sql.strip().upper().startswith("SELECT"):
                results = []
                while self.query.next():
                    row = []
                    for i in range(self.query.record().count()):
                        value = self.query.value(i)
                        if isinstance(value, QByteArray):
                            value = bytes(value)
                        row.append(value)
                    results.append(row)
                return results
            else:
                return True
        except Exception as e:
            print(f"SQL error: {e}")
            return False