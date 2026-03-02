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

    def operation_sql(self, sql, params=None):
        if params:
            result = self.query.prepare(sql)
            if not result:
                return self.query.lastError().text()
            for param in params:
                self.query.addBindValue(param)
            result = self.query.exec_()
        else:
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


    # 在 sq_db.py 中添加测试函数
    def export_photo(self, name):
        self.operation_sql(f"SELECT 照片 FROM student_info WHERE 姓名 = '{name}'")
        if self.query.next():
            blob_data = self.query.value(0)
            if blob_data:
                with open("exported_photo.jpg", "wb") as f:
                    f.write(blob_data)
                print("✅ 图片已导出为 exported_photo.jpg")
            else:
                print("❌ 照片字段为空")






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

    # 插入信息
    ret = db.operation_sql(
        """
        INSERT INTO student_info (ID, 姓名, 年龄, 性别, 学号, 录入时间, 照片)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [None, '张三', 23, '男', '244345544', '2026-02-28-11:15', img]
    )
    print(ret)

    ret = db.select_name("张三")
    print(ret)

    ret =    db.export_photo("张三")






    # #查询
    # ret = db.select_name("张三")
    # print(ret)

    # #删除
    # ret = db.operation_sql(f"delete from student_info where ID = 4")
    # print(ret)