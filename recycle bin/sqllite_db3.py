import sys
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlError
from PyQt5.QtCore import QDate, QTime, QDateTime, QFile, QThread
from PyQt5.QtGui import QPixmap


class MySqlThread(QThread):
    def __init__(self):
        super().__init__()

        self.db = QSqlDatabase.addDatabase("QSQLITE")

    def create_table(self):