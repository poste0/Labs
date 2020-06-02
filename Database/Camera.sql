create table camera (
	ID varchar(36) not null,
	name varchar(255),
    	address varchar(255),
    	height integer,
    	width integer,
    	frameRate integer,
    	userId varchar(36),

    	primary key (ID)
);
alter table CAMERA add constraint FK_LABADATABASE_CAMERA_ON_USERID foreign key (USERID) references LABA_USER(ID);
