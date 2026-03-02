

"""
     自定义类，通过pyqt5的数据库接口实现数据库以下操作
     create_table:  重新创建一个表并添加id(主键)，姓名，年龄，性别，学号，录入时间，照片字段。
     search_table:  能通过姓名匹配出对应的姓名，年龄，性别，学号，录入时间，照片字段内容
     add_tabel： 能将名，年龄，性别，学号，录入时间，照片加入数据库
"""

from PyQt5.QtSql import QSqlQuery, QSqlDatabase


class MySqlite:
    def __init__(self):
        """
        连接数据库
        """
        try:
            # 添加数据库连接
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            # 设置数据库名字
            self.database.setDatabaseName("student.db")
            # 打开数据库
            self.database.open()
            self.query = QSqlQuery()

        except Exception as e:
            print(f"无法建立连接，error:{e}")

    def operation_sql(self, sql:str):
        """执行sql语句"""

        result = self.query.exec_(sql)

        if result:
            return True
        else:

            return self.query.lastError().text()

    def selectData(self, sql):
        """
        查询
        :return:  {姓名：xx,年龄：xx,性别：xx,学号：xx,录入时间：xxx.照片：xxx}
        """
        dic = {}
        result = self.operation_sql(sql)
        if result:
            # 取查询结果的字段信息（列名、数据类型等）
            self.result = self.query.record()
            while self.query.next(): # 迭代每一行数据
                for i in range(self.result.count()): # 遍历每一个字段
                    print(str(self.query.value(i)))
        else:
            raise Exception(result)

    def select_name(self,name:str):
        """
        通过姓名查询
        :param name: 姓名
        :return:  {姓名：xx,年龄：xx,性别：xx,学号：xx,录入时间：xxx.照片：xxx}
        """
        # result = self.query.exec_(f"select * from student_info  WHERE 姓名 = {name}")

        dic = {}
        sql_ = f"select * from student_info  WHERE 姓名 = '{name}'"
        result = self.operation_sql(f"select * from student_info  WHERE 姓名 = '{name}'")
        if result:
            # 取查询结果的字段信息（列名、数据类型等）
            self.result = self.query.record()
            while self.query.next():  # 迭代每一行数据
                for i in range(self.result.count()):  # 遍历每一个字段
                    print(str(self.query.value(i)))
        else:
            raise Exception(result)



if __name__ == '__main__':
    db = MySqlite()
    # 建表 主键：用来标识每一行数据
    ret = db.operation_sql("""

    create table IF  NOT EXISTS   student_info(
                ID integer  primary key AUTOINCREMENT, 
                姓名 text,
                年龄 int,
                性别 text,
                学号 text,
                录入时间 text,
                照片 BLOB
    )
    """)
    print(ret)
    # 将图片转为二进制
    with open(r'C:\Users\33122\Desktop\face_sys\dataset\images\test\ldh01.png', 'rb') as f:
        img = f.read()

    hex_data = img.hex()
    #往表中添加数据
    ret = db.operation_sql(f"""
        insert into student_info values(
                null,
                '李四',
                23,
                '男',
                '244345544',
                "2026-02-28-11:15",
                "123"
        )
        """)
    print(ret)

    # db.selectData("select * from student_info")
    db.select_name("张三")
    #
    # ret = db.operation_sql(f"update student_info set 照片 = X{hex_data} where id = 1")
    # print(ret)


