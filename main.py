import sys
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel
from datetime import datetime
import os


class SqliteDatabaseManager:
    def __init__(self, db_name="FaceDatabase.db"):
        self.db_name = db_name
        self.connection = None
        self.create_connection()
        self.create_table()

    def create_connection(self):
        """创建数据库连接"""
        try:
            # 确保数据库目录存在
            db_dir = os.path.join(os.getcwd(), "static", "sqlite")
            os.makedirs(db_dir, exist_ok=True)
            self.db_path = os.path.join(db_dir, self.db_name)

            # 使用PyQt的SQL接口连接数据库
            self.db = QSqlDatabase.addDatabase("QSQLITE")
            self.db.setDatabaseName(self.db_path)

            if not self.db.open():
                print(f"数据库连接失败: {self.db.lastError().text()}")
                return False
            else:
                print(f"成功连接到数据库: {self.db_path}")
                return True
        except Exception as e:
            print(f"数据库连接异常: {e}")
            return False

    def create_table(self):
        """重新创建表并添加id(主键)，姓名，时间，性别，学号，录入时间，照片字段"""
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
                学号 TEXT,
                录入时间 TEXT,
                照片 BLOB
            )
            """
            if not query.exec_(create_sql):
                print(f"创建表失败: {query.lastError().text()}")
                return False
            else:
                print("数据表创建成功")
                return True
        except Exception as e:
            print(f"创建表异常: {e}")
            return False

    def add_record(self, name, time, gender, student_id, photo_path=None):
        """将姓名、时间、性别、学号、录入时间、照片加入数据库"""
        try:
            query = QSqlQuery(self.db)
            # 获取当前时间作为录入时间
            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 如果提供了照片路径，则读取照片数据
            photo_blob = None
            if photo_path and os.path.exists(photo_path):
                with open(photo_path, 'rb') as f:
                    photo_blob = f.read()

            # 准备SQL语句
            insert_sql = """
            INSERT INTO facedatabase (姓名, 时间, 性别, 学号, 录入时间, 照片)
            VALUES (:name, :time, :gender, :student_id, :entry_time, :photo)
            """
            query.prepare(insert_sql)
            query.bindValue(":name", name)
            query.bindValue(":time", time)
            query.bindValue(":gender", gender)
            query.bindValue(":student_id", student_id)
            query.bindValue(":entry_time", entry_time)
            query.bindValue(":photo", photo_blob if photo_blob else None)

            if not query.exec_():
                print(f"添加记录失败: {query.lastError().text()}")
                return False
            else:
                print("记录添加成功")
                return True
        except Exception as e:
            print(f"添加记录异常: {e}")
            return False

    def search_records(self, name_pattern):
        """通过姓名匹配出对应的姓名，时间，性别，学号，录入时间，照片字段内容"""
        try:
            query = QSqlQuery(self.db)
            # 使用LIKE进行模糊匹配
            search_sql = """
            SELECT 姓名, 时间, 性别, 学号, 录入时间, 照片, id
            FROM facedatabase
            WHERE 姓名 LIKE :pattern
            ORDER BY id DESC
            """
            query.prepare(search_sql)
            query.bindValue(":pattern", f"%{name_pattern}%")

            if not query.exec_():
                print(f"搜索失败: {query.lastError().text()}")
                return []

            results = []
            while query.next():
                record = {
                    '姓名': query.value(0),
                    '时间': query.value(1),
                    '性别': query.value(2),
                    '学号': query.value(3),
                    '录入时间': query.value(4),
                    '照片': query.value(5),  # BLOB数据
                    'id': query.value(6)
                }
                results.append(record)

            return results
        except Exception as e:
            print(f"搜索异常: {e}")
            return []

    def get_all_records(self):
        """获取所有记录"""
        try:
            query = QSqlQuery(self.db)
            select_sql = """
            SELECT 姓名, 时间, 性别, 学号, 录入时间, 照片, id
            FROM facedatabase
            ORDER BY id DESC
            """
            query.exec_(select_sql)

            results = []
            while query.next():
                record = {
                    '姓名': query.value(0),
                    '时间': query.value(1),
                    '性别': query.value(2),
                    '学号': query.value(3),
                    '录入时间': query.value(4),
                    '照片': query.value(5),
                    'id': query.value(6)
                }
                results.append(record)

            return results
        except Exception as e:
            print(f"获取所有记录异常: {e}")
            return []

    def delete_record_by_id(self, record_id):
        """根据ID删除记录"""
        try:
            query = QSqlQuery(self.db)
            delete_sql = "DELETE FROM facedatabase WHERE id = :id"
            query.prepare(delete_sql)
            query.bindValue(":id", record_id)

            if not query.exec_():
                print(f"删除记录失败: {query.lastError().text()}")
                return False
            else:
                print(f"记录ID {record_id} 删除成功")
                return True
        except Exception as e:
            print(f"删除记录异常: {e}")
            return False

    def close_connection(self):
        """关闭数据库连接"""
        if self.db.isOpen():
            self.db.close()
            print("数据库连接已关闭")


class FaceRecognitionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db_manager = SqliteDatabaseManager()
        self.current_photo_path = None
        self.init_ui()

    def init_ui(self):
        # 导入UI类
        from PyQt5 import QtCore, QtGui, QtWidgets

        class Ui_Form(object):
            def setupUi(self, Form):
                Form.setObjectName("Form")
                Form.resize(1173, 904)
                Form.setStyleSheet("""
                QWidget {
                    font-family: "Microsoft YaHei", Arial, sans-serif;
                    font-size: 14px;
                    color: #192a56;
                    background-color: #e8f4fc;
                }

                QLabel#label_28 {
                    font-size: 24px;
                    font-weight: bold;
                    color: #0a5cad;
                    padding: 10px 0;
                    border-bottom: 2px solid #1890ff;
                    margin-bottom: 10px;
                    background-color: #ffffff;
                    border-radius: 8px 8px 0 0;
                }

                QGroupBox#groupBox {
                    font-size: 16px;
                    font-weight: bold;
                    color: #0a5cad;
                    border: 1px solid #b3d9f2;
                    border-radius: 8px;
                    padding-top: 20px;
                    margin: 5px;
                    background-color: #ffffff;
                }

                QGroupBox#groupBox::title {
                    subcontrol-origin: margin;
                    left: 20px;
                    padding: 0 10px 0 10px;
                    color: #0a5cad;
                }

                QPushButton#pushButton_sb_3, QPushButton#pushButton_25, 
                QPushButton#pushButton_gl_3, QPushButton#pushButton_lr_3 {
                    font-size: 16px;
                    padding: 15px 0;
                    border-radius: 8px;
                    border: none;
                    background-color: #1890ff;
                    color: white;
                }

                QPushButton#pushButton_sb_3:hover, QPushButton#pushButton_25:hover, 
                QPushButton#pushButton_gl_3:hover {
                    background-color: #0f6ebb;
                }

                QPushButton#pushButton_lr_3 {
                    background-color: #ff4d6d;
                }

                QPushButton#pushButton_lr_3:hover {
                    background-color: #d63756;
                }

                QGroupBox#groupBox_2, QGroupBox#groupBox_3, QGroupBox#groupBox_4 {
                    font-size: 16px;
                    font-weight: bold;
                    color: #0a5cad;
                    border: 1px solid #b3d9f2;
                    border-radius: 8px;
                    padding-top: 20px;
                    margin: 5px;
                    background-color: #ffffff;
                }

                QGroupBox#groupBox_2::title, QGroupBox#groupBox_3::title, QGroupBox#groupBox_4::title {
                    subcontrol-origin: margin;
                    left: 20px;
                    padding: 0 10px 0 10px;
                    color: #0a5cad;
                }

                QFrame#frame, QFrame#frame_2 {
                    border: 1px solid #b3d9f2;
                    border-radius: 8px;
                    background-color: #ffffff;
                    padding: 10px;
                    box-shadow: 0 2px 4px rgba(24, 144, 255, 0.1);
                }

                QLabel#label_29, QLabel#label_27 {
                    font-size: 16px;
                    font-weight: bold;
                    color: #0a5cad;
                    padding: 10px 0;
                    border-bottom: 1px solid #b3d9f2;
                    margin-bottom: 10px;
                }

                QPushButton:not(#pushButton_sb_3):not(#pushButton_25):not(#pushButton_gl_3):not(#pushButton_lr_3) {
                    padding: 8px 15px;
                    border-radius: 6px;
                    border: 1px solid #1890ff;
                    background-color: #ffffff;
                    color: #0a5cad;
                }

                QPushButton:not(#pushButton_sb_3):not(#pushButton_25):not(#pushButton_gl_3):not(#pushButton_lr_3):hover {
                    background-color: #f0f7ff;
                    border-color: #0f6ebb;
                }

                QPushButton#pushButton_26 {
                    background-color: #52c41a;
                    color: white;
                    border: none;
                }

                QPushButton#pushButton_26:hover {
                    background-color: #389e0d;
                }

                QLineEdit {
                    padding: 8px 10px;
                    border: 1px solid #b3d9f2;
                    border-radius: 6px;
                    background-color: #ffffff;
                    color: #0a5cad;
                }

                QLineEdit:focus {
                    border-color: #1890ff;
                    outline: none;
                    box-shadow: 0 0 4px rgba(24, 144, 255, 0.2);
                }

                QRadioButton {
                    padding: 5px;
                    color: #0a5cad;
                }

                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                }

                QRadioButton::indicator:checked {
                    background-color: #1890ff;
                    border-color: #0f6ebb;
                }

                QPlainTextEdit#plainTextEdit_2 {
                    border: 1px solid #b3d9f2;
                    border-radius: 6px;
                    padding: 10px;
                    background-color: #ffffff;
                    color: #0a5cad;
                }

                QPlainTextEdit#plainTextEdit_2:focus {
                    border-color: #1890ff;
                    outline: none;
                }

                QTableView#tableView_2 {
                    border: 1px solid #b3d9f2;
                    border-radius: 6px;
                    background-color: #ffffff;
                    gridline-color: #e8f4fc;
                    color: #0a5cad;
                }

                QTableView#tableView_2::item {
                    padding: 5px;
                }

                QTableView#tableView_2::item:selected {
                    background-color: #e6f7ff;
                    color: #0a5cad;
                }

                QHeaderView::section {
                    background-color: #f0f7ff;
                    border: 1px solid #b3d9f2;
                    padding: 8px;
                    font-weight: bold;
                    color: #0a5cad;
                }

                QLabel#label_15, QLabel#label_30, QLabel#label_24 {
                    border: 1px solid #b3d9f2;
                    border-radius: 6px;
                    background-color: #f0f7ff;
                }
                """)

                self.verticalLayout_14 = QtWidgets.QVBoxLayout(Form)
                self.verticalLayout_14.setObjectName("verticalLayout_14")
                self.label_28 = QtWidgets.QLabel(parent=Form)
                self.label_28.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.label_28.setObjectName("label_28")
                self.verticalLayout_14.addWidget(self.label_28)
                self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
                self.horizontalLayout_10.setObjectName("horizontalLayout_10")
                self.groupBox = QtWidgets.QGroupBox(parent=Form)
                self.groupBox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.groupBox.setFlat(False)
                self.groupBox.setCheckable(False)
                self.groupBox.setObjectName("groupBox")
                self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
                self.verticalLayout.setSpacing(150)
                self.verticalLayout.setObjectName("verticalLayout")
                self.verticalLayout_15 = QtWidgets.QVBoxLayout()
                self.verticalLayout_15.setObjectName("verticalLayout_15")
                self.pushButton_sb_3 = QtWidgets.QPushButton(parent=self.groupBox)
                self.pushButton_sb_3.setAutoDefault(False)
                self.pushButton_sb_3.setDefault(False)
                self.pushButton_sb_3.setFlat(False)
                self.pushButton_sb_3.setObjectName("pushButton_sb_3")
                self.verticalLayout_15.addWidget(self.pushButton_sb_3)
                self.pushButton_25 = QtWidgets.QPushButton(parent=self.groupBox)
                self.pushButton_25.setObjectName("pushButton_25")
                self.verticalLayout_15.addWidget(self.pushButton_25)
                self.pushButton_gl_3 = QtWidgets.QPushButton(parent=self.groupBox)
                self.pushButton_gl_3.setObjectName("pushButton_gl_3")
                self.verticalLayout_15.addWidget(self.pushButton_gl_3)
                self.pushButton_lr_3 = QtWidgets.QPushButton(parent=self.groupBox)
                self.pushButton_lr_3.setObjectName("pushButton_lr_3")
                self.verticalLayout_15.addWidget(self.pushButton_lr_3)
                self.verticalLayout_15.setStretch(0, 2)
                self.verticalLayout_15.setStretch(1, 2)
                self.verticalLayout_15.setStretch(2, 2)
                self.verticalLayout_15.setStretch(3, 2)
                self.verticalLayout.addLayout(self.verticalLayout_15)
                self.horizontalLayout_10.addWidget(self.groupBox)
                self.stackedWidget_2 = QtWidgets.QStackedWidget(parent=Form)
                self.stackedWidget_2.setObjectName("stackedWidget_2")
                self.page_1 = QtWidgets.QWidget()
                self.page_1.setObjectName("page_1")
                self.groupBox_2 = QtWidgets.QGroupBox(parent=self.page_1)
                self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 891, 761))
                self.groupBox_2.setObjectName("groupBox_2")
                self.horizontalLayoutWidget_3 = QtWidgets.QWidget(parent=self.groupBox_2)
                self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 20, 861, 731))
                self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
                self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
                self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
                self.horizontalLayout_11.setObjectName("horizontalLayout_11")
                self.verticalLayout_9 = QtWidgets.QVBoxLayout()
                self.verticalLayout_9.setObjectName("verticalLayout_9")
                self.label_15 = QtWidgets.QLabel(parent=self.horizontalLayoutWidget_3)
                self.label_15.setText("")
                self.label_15.setPixmap(QtGui.QPixmap())
                self.label_15.setScaledContents(True)
                self.label_15.setObjectName("label_15")
                self.verticalLayout_9.addWidget(self.label_15)
                self.label_16 = QtWidgets.QLabel(parent=self.horizontalLayoutWidget_3)
                self.label_16.setObjectName("label_16")
                self.verticalLayout_9.addWidget(self.label_16)
                self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
                self.horizontalLayout_12.setObjectName("horizontalLayout_12")
                self.pushButton_14 = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget_3)
                self.pushButton_14.setObjectName("pushButton_14")
                self.horizontalLayout_12.addWidget(self.pushButton_14)
                self.lineEdit_7 = QtWidgets.QLineEdit(parent=self.horizontalLayoutWidget_3)
                self.lineEdit_7.setObjectName("lineEdit_7")
                self.horizontalLayout_12.addWidget(self.lineEdit_7)
                self.verticalLayout_9.addLayout(self.horizontalLayout_12)
                self.label_17 = QtWidgets.QLabel(parent=self.horizontalLayoutWidget_3)
                self.label_17.setObjectName("label_17")
                self.verticalLayout_9.addWidget(self.label_17)
                self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
                self.horizontalLayout_13.setObjectName("horizontalLayout_13")
                self.pushButton_15 = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget_3)
                self.pushButton_15.setObjectName("pushButton_15")
                self.horizontalLayout_13.addWidget(self.pushButton_15)
                self.pushButton_16 = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget_3)
                self.pushButton_16.setObjectName("pushButton_16")
                self.horizontalLayout_13.addWidget(self.pushButton_16)
                self.pushButton_17 = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget_3)
                self.pushButton_17.setObjectName("pushButton_17")
                self.horizontalLayout_13.addWidget(self.pushButton_17)
                self.verticalLayout_9.addLayout(self.horizontalLayout_13)
                self.verticalLayout_9.setStretch(0, 6)
                self.verticalLayout_9.setStretch(1, 1)
                self.verticalLayout_9.setStretch(2, 1)
                self.verticalLayout_9.setStretch(3, 1)
                self.verticalLayout_9.setStretch(4, 1)
                self.horizontalLayout_11.addLayout(self.verticalLayout_9)
                self.frame = QtWidgets.QFrame(parent=self.horizontalLayoutWidget_3)
                self.frame.setFrameShape(QtWidgets.QFrame.Shape.Box)
                self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
                self.frame.setObjectName("frame")
                self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
                self.verticalLayout_2.setObjectName("verticalLayout_2")
                self.verticalLayout_16 = QtWidgets.QVBoxLayout()
                self.verticalLayout_16.setSpacing(15)
                self.verticalLayout_16.setObjectName("verticalLayout_16")
                self.label_29 = QtWidgets.QLabel(parent=self.frame)
                self.label_29.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.label_29.setObjectName("label_29")
                self.verticalLayout_16.addWidget(self.label_29)
                self.label_30 = QtWidgets.QLabel(parent=self.frame)
                self.label_30.setText("")
                self.label_30.setPixmap(QtGui.QPixmap())
                self.label_30.setScaledContents(True)
                self.label_30.setObjectName("label_30")
                self.verticalLayout_16.addWidget(self.label_30)
                self.formLayout_3 = QtWidgets.QFormLayout()
                self.formLayout_3.setVerticalSpacing(60)
                self.formLayout_3.setObjectName("formLayout_3")
                self.label_31 = QtWidgets.QLabel(parent=self.frame)
                self.label_31.setObjectName("label_31")
                self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_31)
                self.label_32 = QtWidgets.QLabel(parent=self.frame)
                self.label_32.setObjectName("label_32")
                self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_32)
                self.label_33 = QtWidgets.QLabel(parent=self.frame)
                self.label_33.setObjectName("label_33")
                self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_33)
                self.lineEdit_13 = QtWidgets.QLineEdit(parent=self.frame)
                self.lineEdit_13.setObjectName("lineEdit_13")
                self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.lineEdit_13)
                self.lineEdit_14 = QtWidgets.QLineEdit(parent=self.frame)
                self.lineEdit_14.setObjectName("lineEdit_14")
                self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.lineEdit_14)
                self.lineEdit_15 = QtWidgets.QLineEdit(parent=self.frame)
                self.lineEdit_15.setObjectName("lineEdit_15")
                self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.lineEdit_15)
                self.verticalLayout_16.addLayout(self.formLayout_3)
                self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
                self.horizontalLayout_19.setContentsMargins(-1, 0, -1, 60)
                self.horizontalLayout_19.setObjectName("horizontalLayout_19")
                self.label_34 = QtWidgets.QLabel(parent=self.frame)
                self.label_34.setObjectName("label_34")
                self.horizontalLayout_19.addWidget(self.label_34)
                self.radioButton_5 = QtWidgets.QRadioButton(parent=self.frame)
                self.radioButton_5.setChecked(True)
                self.radioButton_5.setObjectName("radioButton_5")
                self.horizontalLayout_19.addWidget(self.radioButton_5)
                self.radioButton_6 = QtWidgets.QRadioButton(parent=self.frame)
                self.radioButton_6.setCheckable(True)
                self.radioButton_6.setChecked(False)
                self.radioButton_6.setObjectName("radioButton_6")
                self.horizontalLayout_19.addWidget(self.radioButton_6)
                self.verticalLayout_16.addLayout(self.horizontalLayout_19)
                self.pushButton_26 = QtWidgets.QPushButton(parent=self.frame)
                self.pushButton_26.setObjectName("pushButton_26")
                self.verticalLayout_16.addWidget(self.pushButton_26)
                self.verticalLayout_16.setStretch(0, 1)
                self.verticalLayout_16.setStretch(1, 3)
                self.verticalLayout_16.setStretch(2, 3)
                self.verticalLayout_16.setStretch(3, 2)
                self.verticalLayout_16.setStretch(4, 1)
                self.verticalLayout_2.addLayout(self.verticalLayout_16)
                self.horizontalLayout_11.addWidget(self.frame)
                self.horizontalLayout_11.setStretch(0, 7)
                self.horizontalLayout_11.setStretch(1, 3)
                self.stackedWidget_2.addWidget(self.page_1)
                self.page_2 = QtWidgets.QWidget()
                self.page_2.setObjectName("page_2")
                self.groupBox_3 = QtWidgets.QGroupBox(parent=self.page_2)
                self.groupBox_3.setGeometry(QtCore.QRect(10, 10, 891, 761))
                self.groupBox_3.setObjectName("groupBox_3")
                self.horizontalLayoutWidget_7 = QtWidgets.QWidget(parent=self.groupBox_3)
                self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(10, 20, 871, 731))
                self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
                self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
                self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
                self.horizontalLayout_15.setObjectName("horizontalLayout_15")
                self.verticalLayout_11 = QtWidgets.QVBoxLayout()
                self.verticalLayout_11.setObjectName("verticalLayout_11")
                self.label_24 = QtWidgets.QLabel(parent=self.horizontalLayoutWidget_7)
                self.label_24.setText("")
                self.label_24.setPixmap(QtGui.QPixmap())
                self.label_24.setScaledContents(True)
                self.label_24.setObjectName("label_24")
                self.verticalLayout_11.addWidget(self.label_24)
                self.label_25 = QtWidgets.QLabel(parent=self.horizontalLayoutWidget_7)
                self.label_25.setObjectName("label_25")
                self.verticalLayout_11.addWidget(self.label_25)
                self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
                self.horizontalLayout_16.setObjectName("horizontalLayout_16")
                self.pushButton_19 = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget_7)
                self.pushButton_19.setObjectName("pushButton_19")
                self.horizontalLayout_16.addWidget(self.pushButton_19)
                self.lineEdit_11 = QtWidgets.QLineEdit(parent=self.horizontalLayoutWidget_7)
                self.lineEdit_11.setObjectName("lineEdit_11")
                self.horizontalLayout_16.addWidget(self.lineEdit_11)
                self.verticalLayout_11.addLayout(self.horizontalLayout_16)
                self.label_26 = QtWidgets.QLabel(parent=self.horizontalLayoutWidget_7)
                self.label_26.setObjectName("label_26")
                self.verticalLayout_11.addWidget(self.label_26)
                self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
                self.horizontalLayout_17.setObjectName("horizontalLayout_17")
                self.pushButton_20 = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget_7)
                self.pushButton_20.setObjectName("pushButton_20")
                self.horizontalLayout_17.addWidget(self.pushButton_20)
                self.pushButton_21 = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget_7)
                self.pushButton_21.setObjectName("pushButton_21")
                self.horizontalLayout_17.addWidget(self.pushButton_21)
                self.verticalLayout_11.addLayout(self.horizontalLayout_17)
                self.verticalLayout_11.setStretch(0, 6)
                self.verticalLayout_11.setStretch(1, 1)
                self.verticalLayout_11.setStretch(2, 1)
                self.verticalLayout_11.setStretch(3, 1)
                self.verticalLayout_11.setStretch(4, 1)
                self.horizontalLayout_15.addLayout(self.verticalLayout_11)
                self.frame_2 = QtWidgets.QFrame(parent=self.horizontalLayoutWidget_7)
                self.frame_2.setFrameShape(QtWidgets.QFrame.Shape.Box)
                self.frame_2.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
                self.frame_2.setObjectName("frame_2")
                self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_2)
                self.verticalLayout_3.setObjectName("verticalLayout_3")
                self.verticalLayout_12 = QtWidgets.QVBoxLayout()
                self.verticalLayout_12.setObjectName("verticalLayout_12")
                self.label_27 = QtWidgets.QLabel(parent=self.frame_2)
                self.label_27.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.label_27.setObjectName("label_27")
                self.verticalLayout_12.addWidget(self.label_27)
                self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(parent=self.frame_2)
                self.plainTextEdit_2.setObjectName("plainTextEdit_2")
                self.verticalLayout_12.addWidget(self.plainTextEdit_2)
                self.verticalLayout_12.setStretch(0, 2)
                self.verticalLayout_12.setStretch(1, 8)
                self.verticalLayout_3.addLayout(self.verticalLayout_12)
                self.horizontalLayout_15.addWidget(self.frame_2)
                self.horizontalLayout_15.setStretch(0, 7)
                self.stackedWidget_2.addWidget(self.page_2)
                self.page_3 = QtWidgets.QWidget()
                self.page_3.setObjectName("page_3")
                self.groupBox_4 = QtWidgets.QGroupBox(parent=self.page_3)
                self.groupBox_4.setGeometry(QtCore.QRect(10, 10, 891, 761))
                self.groupBox_4.setObjectName("groupBox_4")
                self.verticalLayoutWidget_6 = QtWidgets.QWidget(parent=self.groupBox_4)
                self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(10, 20, 871, 731))
                self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
                self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
                self.verticalLayout_13.setContentsMargins(0, 0, 0, 0)
                self.verticalLayout_13.setSpacing(50)
                self.verticalLayout_13.setObjectName("verticalLayout_13")
                self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
                self.horizontalLayout_18.setContentsMargins(-1, 20, -1, 0)
                self.horizontalLayout_18.setSpacing(20)
                self.horizontalLayout_18.setObjectName("horizontalLayout_18")
                self.lineEdit_12 = QtWidgets.QLineEdit(parent=self.verticalLayoutWidget_6)
                self.lineEdit_12.setObjectName("lineEdit_12")
                self.horizontalLayout_18.addWidget(self.lineEdit_12)
                self.pushButton_22 = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_6)
                self.pushButton_22.setObjectName("pushButton_22")
                self.horizontalLayout_18.addWidget(self.pushButton_22)
                self.pushButton_23 = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_6)
                self.pushButton_23.setObjectName("pushButton_23")
                self.horizontalLayout_18.addWidget(self.pushButton_23)
                self.pushButton_24 = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_6)
                self.pushButton_24.setObjectName("pushButton_24")
                self.horizontalLayout_18.addWidget(self.pushButton_24)
                self.horizontalLayout_18.setStretch(0, 2)
                self.horizontalLayout_18.setStretch(1, 1)
                self.horizontalLayout_18.setStretch(2, 1)
                self.horizontalLayout_18.setStretch(3, 1)
                self.verticalLayout_13.addLayout(self.horizontalLayout_18)
                self.tableView_2 = QtWidgets.QTableView(parent=self.verticalLayoutWidget_6)
                self.tableView_2.setObjectName("tableView_2")
                self.verticalLayout_13.addWidget(self.tableView_2)
                self.verticalLayout_13.setStretch(0, 2)
                self.verticalLayout_13.setStretch(1, 6)
                self.stackedWidget_2.addWidget(self.page_3)
                self.horizontalLayout_10.addWidget(self.stackedWidget_2)
                self.horizontalLayout_10.setStretch(0, 2)
                self.horizontalLayout_10.setStretch(1, 8)
                self.verticalLayout_14.addLayout(self.horizontalLayout_10)
                self.verticalLayout_14.setStretch(0, 1)
                self.verticalLayout_14.setStretch(1, 9)

                self.retranslateUi(Form)
                self.stackedWidget_2.setCurrentIndex(0)
                QtCore.QMetaObject.connectSlotsByName(Form)

            def retranslateUi(self, Form):
                _translate = QtCore.QCoreApplication.translate
                Form.setWindowTitle(_translate("Form", "人脸识别系统"))
                self.label_28.setText(_translate("Form", "人脸识别系统"))
                self.groupBox.setTitle(_translate("Form", "功能选项"))
                self.pushButton_sb_3.setText(_translate("Form", "人脸信息录入"))
                self.pushButton_25.setText(_translate("Form", "人脸识别"))
                self.pushButton_gl_3.setText(_translate("Form", "数据库管理"))
                self.pushButton_lr_3.setText(_translate("Form", "退出"))
                self.groupBox_2.setTitle(_translate("Form", "人脸信息录入"))
                self.label_16.setText(_translate("Form", "方式一：图片"))
                self.pushButton_14.setText(_translate("Form", "上传图片"))
                self.label_17.setText(_translate("Form", "方式二：摄像头"))
                self.pushButton_15.setText(_translate("Form", "开启摄像头"))
                self.pushButton_16.setText(_translate("Form", "拍照"))
                self.pushButton_17.setText(_translate("Form", "关闭摄像头"))
                self.label_29.setText(_translate("Form", "人脸信息"))
                self.label_31.setText(_translate("Form", "姓名："))
                self.label_32.setText(_translate("Form", "年龄："))
                self.label_33.setText(_translate("Form", "学号："))
                self.label_34.setText(_translate("Form", "性别："))
                self.radioButton_5.setText(_translate("Form", "男"))
                self.radioButton_6.setText(_translate("Form", "女"))
                self.pushButton_26.setText(_translate("Form", "保存信息"))
                self.groupBox_3.setTitle(_translate("Form", "人脸识别"))
                self.label_25.setText(_translate("Form", "方式一：图片"))
                self.pushButton_19.setText(_translate("Form", "上传图片"))
                self.label_26.setText(_translate("Form", "方式二：摄像头"))
                self.pushButton_20.setText(_translate("Form", "开启摄像头"))
                self.pushButton_21.setText(_translate("Form", "关闭摄像头"))
                self.label_27.setText(_translate("Form", "识别结果"))
                self.groupBox_4.setTitle(_translate("Form", "数据库管理"))
                self.pushButton_22.setText(_translate("Form", "查询"))
                self.pushButton_23.setText(_translate("Form", "删除"))
                self.pushButton_24.setText(_translate("Form", "刷新"))

        # 设置UI
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 设置窗口标题
        self.setWindowTitle("人脸识别系统 - 数据库管理")

        # 连接信号和槽
        self.ui.pushButton_sb_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))  # 人脸信息录入
        self.ui.pushButton_25.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))  # 人脸识别
        self.ui.pushButton_gl_3.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(2))  # 数据库管理
        self.ui.pushButton_lr_3.clicked.connect(self.close)  # 退出

        # 人脸录入页面按钮连接
        self.ui.pushButton_14.clicked.connect(self.upload_image)  # 上传图片
        self.ui.pushButton_26.clicked.connect(self.save_person_info)  # 保存信息

        # 数据库管理页面按钮连接
        self.ui.pushButton_22.clicked.connect(self.search_database)  # 查询
        self.ui.pushButton_23.clicked.connect(self.delete_selected)  # 删除
        self.ui.pushButton_24.clicked.connect(self.refresh_table)  # 刷新

        # 初始化数据库表格
        self.refresh_table()

    def upload_image(self):
        """上传图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.current_photo_path = file_path
            self.ui.lineEdit_7.setText(file_path)
            # 显示图片预览
            pixmap = QtGui.QPixmap(file_path)
            self.ui.label_15.setPixmap(pixmap.scaled(
                self.ui.label_15.width(),
                self.ui.label_15.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            ))

    def save_person_info(self):
        """保存人员信息到数据库"""
        name = self.ui.lineEdit_13.text().strip()
        time = self.ui.lineEdit_14.text().strip()
        student_id = self.ui.lineEdit_15.text().strip()

        if not name:
            QMessageBox.warning(self, "警告", "姓名不能为空！")
            return

        # 获取性别
        gender = "男" if self.ui.radioButton_5.isChecked() else "女"

        # 调用数据库管理器添加记录
        success = self.db_manager.add_record(
            name=name,
            time=time,
            gender=gender,
            student_id=student_id,
            photo_path=self.current_photo_path
        )

        if success:
            QMessageBox.information(self, "成功", "人员信息保存成功！")
            # 清空输入框
            self.ui.lineEdit_13.clear()
            self.ui.lineEdit_14.clear()
            self.ui.lineEdit_15.clear()
            self.ui.lineEdit_7.clear()
            self.ui.label_15.clear()
            self.current_photo_path = None
        else:
            QMessageBox.critical(self, "错误", "保存人员信息失败！")

    def search_database(self):
        """搜索数据库"""
        keyword = self.ui.lineEdit_12.text().strip()
        if not keyword:
            QMessageBox.warning(self, "警告", "请输入搜索关键词！")
            return

        results = self.db_manager.search_records(keyword)

        if not results:
            QMessageBox.information(self, "提示", f"未找到匹配 '{keyword}' 的记录！")
            return

        # 显示搜索结果
        self.display_results_in_table(results)

    def display_results_in_table(self, results):
        """在表格中显示结果"""
        # 设置表格列数和标题
        self.ui.tableView_2.setColumnCount(6)
        self.ui.tableView_2.setHorizontalHeaderLabels([
            "姓名", "时间", "性别", "学号", "录入时间", "ID"
        ])

        # 设置表格行数
        self.ui.tableView_2.setRowCount(len(results))

        # 填充数据
        for row_idx, record in enumerate(results):
            self.ui.tableView_2.setItem(row_idx, 0, QTableWidgetItem(str(record['姓名'])))
            self.ui.tableView_2.setItem(row_idx, 1, QTableWidgetItem(str(record['时间']) if record['时间'] else ""))
            self.ui.tableView_2.setItem(row_idx, 2, QTableWidgetItem(str(record['性别']) if record['性别'] else ""))
            self.ui.tableView_2.setItem(row_idx, 3, QTableWidgetItem(str(record['学号']) if record['学号'] else ""))
            self.ui.tableView_2.setItem(row_idx, 4,
                                        QTableWidgetItem(str(record['录入时间']) if record['录入时间'] else ""))
            self.ui.tableView_2.setItem(row_idx, 5, QTableWidgetItem(str(record['id'])))

    def delete_selected(self):
        """删除选中行"""
        selected_rows = self.ui.tableView_2.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请选择要删除的记录！")
            return

        # 获取选中行的ID
        ids_to_delete = []
        for index in selected_rows:
            id_item = self.ui.tableView_2.item(index.row(), 5)  # ID列是第6列（索引5）
            if id_item:
                ids_to_delete.append(int(id_item.text()))

        # 确认删除
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除选中的 {len(ids_to_delete)} 条记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            success_count = 0
            for record_id in ids_to_delete:
                if self.db_manager.delete_record_by_id(record_id):
                    success_count += 1

            QMessageBox.information(
                self,
                "删除完成",
                f"成功删除 {success_count} 条记录，共选择 {len(ids_to_delete)} 条"
            )
            self.refresh_table()

    def refresh_table(self):
        """刷新表格显示所有记录"""
        all_records = self.db_manager.get_all_records()
        self.display_results_in_table(all_records)

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.db_manager.close_connection()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()