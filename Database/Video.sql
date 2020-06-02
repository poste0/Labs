create table video (
	ID varchar(36) not null,
	name varchar(255),
    	status varchar(255),
    	fileDescriptorId varchar(36),
    	cameraId varchar(36),

    	primary key (ID)
);
alter table VIDEO add constraint FK_LABADATABASE_VIDEO_ON_FILEDESCRIPTORID foreign key (FILEDESCRIPTORID) references FILE_DESCRIPTOR(ID);
alter table VIDEO add constraint FK_LABADATABASE_VIDEO_ON_CAMERAID foreign key (CAMERAID) references CAMERA(ID);
