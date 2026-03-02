from PyQt5.QtSql import QSqlQuery, QSqlDatabase


class MySqlite:
    def __init__(self):
        try:
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            self.database.setDatabaseName("student.db")
            self.database.open()
            self.query = QSqlQuery()

        except Exception as e:
            print(f"无法建立连接，error:{e}")

    def operation_sql(self,sql):
        result = self.query.exec_(sql)

        if result:
            return True
        else:
            return self.query.lastError().text()

    def select_name(self,name):
        result = self.operation_sql(f"select * from student_info WHERE 姓名 = '{name}'")
        if result:
            self.result = self.query.record()
            while self.query.next():
                for i in range(self.result.count()):
                    print(str(self.query.value(i)))
        else:
            raise Exception(result)





if __name__ == '__main__':
    db = MySqlite()

    #创建表
    ret = db.operation_sql("""
    
    create table IF NOT EXISTS student_info(
                ID integer primary key  AUTOINCREMENT,
                姓名 text,
                年龄 int,
                性别 text,
                学号 text,
                录入时间 text,
                照片 BLOB
    )
    """)
    print(ret)

    with open(r'C:\Users\Lenovo\Desktop\HQYJ\Facial Recognition\dataset\images\test\Ziwang_Xu_0001.jpg', 'rb') as f:
        img = f.read()

    #插入信息
    ret = db.operation_sql(f"""
        insert into student_info values(
                null,
                '张三',
                23,
                '男',
                '244345544',
                "2026-02-28-11:15",
                img
        )
        """)
    print(ret)



    # #查询
    # ret = db.select_name("张三")
    # print(ret)

    # #删除
    # ret = db.operation_sql(f"delete from student_info where ID = 4")
    # print(ret)