# 从 PyQt5 的 SQL 模块导入数据库操作类
# QSqlQuery 用于执行 SQL 查询和操作
from PyQt5.QtSql import QSqlQuery, QSqlDatabase
# 从 PyQt5 的 Core 模块导入 QByteArray 类
# QByteArray 用于处理二进制数据（如图片的 BLOB 数据）
from PyQt5.QtCore import QByteArray


# 定义一个名为 MySqlite 的类，用于封装 SQLite 数据库操作
class MySqlite:
    # 类的初始化方法，在创建实例时自动执行
    def __init__(self):
        try:
            # 使用 QSqlDatabase 添加一个 SQLite 类型的数据库连接
            # "QSQLITE" 是 Qt 的 SQLite 数据库驱动名称
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            # 设置数据库文件的名称为 "student.db"
            # 如果文件不存在，SQLite 会自动创建该文件
            self.database.setDatabaseName("student.db")
            # 尝试打开数据库连接
            # 如果成功返回 True，否则返回 False
            self.database.open()
            # 创建一个 QSqlQuery 对象，用于执行 SQL 语句
            # QSqlQuery 提供了执行 SQL 查询和处理结果的方法
            self.query = QSqlQuery()
        except Exception as e:
            # 捕获所有异常并打印错误信息
            # f-string 格式化字符串，将异常信息嵌入到输出中
            print(f"无法建立连接，error:{e}")

    # 定义通用的 SQL 操作方法，支持带参数和不带参数的 SQL 执行
    # sql: SQL 语句字符串
    # params: 可选的参数列表，用于防止 SQL 注入
    def operation_sql(self, sql, params=None):
        try:
            # 判断是否提供了参数
            if params:
                # 准备 SQL 语句（预处理）
                # prepare() 方法用于准备带占位符的 SQL 语句
                # 占位符通常是 :name 或 ? 的形式
                if not self.query.prepare(sql):
                    # 如果准备失败，打印错误信息
                    # lastError().text() 获取最后一次错误的描述文本
                    print(f"SQL 准备失败：{self.query.lastError().text()}")
                    return False
                # 遍历参数列表中的每个参数
                for param in params:
                    # 判断参数是否为字节类型（bytes）
                    # bytes 类型通常用于存储二进制数据，如图片
                    if isinstance(param, bytes):
                        # 将 bytes 转换为 QByteArray 格式
                        # SQLite 的 BLOB 字段需要这种格式
                        self.query.addBindValue(QByteArray(param))
                    else:
                        # 非字节类型直接绑定值
                        # addBindValue() 将参数绑定到 SQL 占位符
                        self.query.addBindValue(param)
                # 执行已准备好的 SQL 语句
                # exec_() 是 PyQt5 中的执行方法（因为 exec 是 Python 关键字）
                if not self.query.exec_():
                    # 如果执行失败，打印错误信息
                    print(f"SQL 执行失败：{self.query.lastError().text()}")
                    return False
            else:
                # 如果没有参数，直接执行 SQL 语句
                # exec_(sql) 方法直接执行不带参数的 SQL
                if not self.query.exec_(sql):
                    # 如果执行失败，打印错误信息
                    print(f"SQL 执行失败：{self.query.lastError().text()}")
                    return False

            # 判断 SQL 语句是否为查询语句（SELECT）
            # strip() 去除首尾空格，upper() 转为大写，startswith() 检查开头
            # 这样可以统一处理不同大小写和格式的 SQL 语句
            if sql.strip().upper().startswith("SELECT"):
                # 如果是查询语句，初始化一个空列表用于存储结果
                results = []
                # 使用 while 循环遍历查询结果集
                # next() 方法将记录指针移动到下一条记录
                # 第一次调用 next() 会移动到第一条记录
                # 如果有记录返回 True，到达结果集末尾返回 False
                while self.query.next():
                    # 初始化一个空列表存储当前行的数据
                    row = []
                    # 遍历当前记录的所有列
                    # record() 获取当前记录的元数据
                    # count() 返回列的数量
                    for i in range(self.query.record().count()):
                        # 获取第 i 列的值
                        # value(i) 返回第 i 列的数据
                        value = self.query.value(i)
                        # 判断值是否为 QByteArray 类型
                        # QByteArray 是 Qt 对二进制数据的封装
                        if isinstance(value, QByteArray):
                            # 将 QByteArray 转换回 Python 的 bytes 类型
                            # bytes() 函数完成类型转换
                            value = bytes(value)
                        # 将处理后的值添加到行列表中
                        row.append(value)
                    # 将完整的行数据添加到结果列表中
                    results.append(row)
                # 返回包含所有查询结果的列表
                # 每个元素代表一行数据，每行数据是一个列表
                return results  # 返回数据列表
            else:
                # 如果不是查询语句（如 INSERT、UPDATE、DELETE）
                # 执行成功则返回 True
                return True   # 非查询语句成功则返回 True

        except Exception as e:
            # 捕获所有异常并打印错误信息
            # str(e) 将异常对象转换为字符串
            print(f"操作 SQL 异常：{str(e)}")
            # 发生异常时返回 False 表示操作失败
            return False


# Python 脚本的主入口判断
# 只有直接运行此文件时才会执行以下代码
# 如果被其他模块导入，则不会执行
if __name__ == '__main__':
    # 实例化 MySqlite 类，创建数据库操作对象
    # 这会自动连接数据库并创建 QSqlQuery 对象
    db = MySqlite()

    # 调用 operation_sql 方法执行创建表的 SQL 语句
    # """ """ 是多行字符串的语法，用于编写跨多行的 SQL 语句
    # CREATE TABLE IF NOT EXISTS 表示如果表不存在才创建
    # student_info 是表名
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
    # 打印操作结果
    # 如果表创建成功，ret 为 True
    # 如果表已存在或其他原因失败，ret 为 False
    print(ret)  # 输出创建表的结果
