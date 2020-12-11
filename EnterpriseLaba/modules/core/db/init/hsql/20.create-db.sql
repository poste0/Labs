-- begin ENTERPRISELABA_THEATRE
alter table ENTERPRISELABA_THEATRE add constraint FK_ENTERPRISELABA_THEATRE_ON_ADMIN foreign key (ADMIN_ID) references SEC_USER(ID)^
create index IDX_ENTERPRISELABA_THEATRE_ON_ADMIN on ENTERPRISELABA_THEATRE (ADMIN_ID)^
-- end ENTERPRISELABA_THEATRE
-- begin ENTERPRISELABA_TICKET
alter table ENTERPRISELABA_TICKET add constraint FK_ENTERPRISELABA_TICKET_ON_SHOW foreign key (SHOW_ID) references ENTERPRISELABA_SHOW(ID)^
create index IDX_ENTERPRISELABA_TICKET_ON_SHOW on ENTERPRISELABA_TICKET (SHOW_ID)^
-- end ENTERPRISELABA_TICKET
-- begin ENTERPRISELABA_AUDITORIUM
alter table ENTERPRISELABA_AUDITORIUM add constraint FK_ENTERPRISELABA_AUDITORIUM_ON_THEATRE foreign key (THEATRE_ID) references ENTERPRISELABA_THEATRE(ID)^
create index IDX_ENTERPRISELABA_AUDITORIUM_ON_THEATRE on ENTERPRISELABA_AUDITORIUM (THEATRE_ID)^
-- end ENTERPRISELABA_AUDITORIUM
-- begin ENTERPRISELABA_SHOW
alter table ENTERPRISELABA_SHOW add constraint FK_ENTERPRISELABA_SHOW_ON_FILM foreign key (FILM_ID) references ENTERPRISELABA_FILM(ID)^
alter table ENTERPRISELABA_SHOW add constraint FK_ENTERPRISELABA_SHOW_ON_THEATRE foreign key (THEATRE_ID) references ENTERPRISELABA_THEATRE(ID)^
alter table ENTERPRISELABA_SHOW add constraint FK_ENTERPRISELABA_SHOW_ON_AUDITORIUM foreign key (AUDITORIUM_ID) references ENTERPRISELABA_AUDITORIUM(ID)^
create index IDX_ENTERPRISELABA_SHOW_ON_FILM on ENTERPRISELABA_SHOW (FILM_ID)^
create index IDX_ENTERPRISELABA_SHOW_ON_THEATRE on ENTERPRISELABA_SHOW (THEATRE_ID)^
create index IDX_ENTERPRISELABA_SHOW_ON_AUDITORIUM on ENTERPRISELABA_SHOW (AUDITORIUM_ID)^
-- end ENTERPRISELABA_SHOW