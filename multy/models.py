from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine


engine = create_engine("mysql://root:admin@127.0.0.1/mygraduationproject", encoding='utf8', echo=False)
Base = declarative_base()


class Reuters10(Base):

    __tablename__ = 'Reuters10'

    id = Column(Integer, primary_key=True)
    sent = Column(String(3000))
    entity1 = Column(String(100))
    entity2 = Column(String(100))
    entity1_idx = Column(Integer)
    entity2_idx = Column(Integer)


if __name__ == '__main__':
    # 创建表 
    Base.metadata.create_all(engine)
