alter table SEC_USER alter column BIRTHDATE rename to BIRTHDATE__U18732 ^
alter table SEC_USER alter column THEATRE_ID rename to THEATRE_ID__U98274 ^
alter table SEC_USER drop constraint FK_SEC_USER_ON_THEATRE ;
drop index IDX_SEC_USER_ON_THEATRE ;
