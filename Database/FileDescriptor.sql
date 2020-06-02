create table FILEDESCRIPTOR(
	ID uuid,
	NAME varchar(255),
	EXTENSION varchar(10),
	SIZE double precision,
	CREATEDATE timestamp,
	video_id varchar(36),

	primary key (ID)
);
alter table FILE_DESCRIPTOR add constraint FK_LABADATABASE_FILE_DESCRIPTOR_ON_VIDEO foreign key (VIDEO_ID) references VIDEO(ID)
