alter table SEC_USER add constraint FK_SEC_USER_ON_THEATRE foreign key (THEATRE_ID) references ENTERPRISELABA_THEATRE(ID);
create index IDX_SEC_USER_ON_THEATRE on SEC_USER (THEATRE_ID);
