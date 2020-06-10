create table ENTERPRISELABA_FILM (
    ID uuid,
    VERSION integer not null,
    CREATE_TS timestamp,
    CREATED_BY varchar(50),
    UPDATE_TS timestamp,
    UPDATED_BY varchar(50),
    DELETE_TS timestamp,
    DELETED_BY varchar(50),
    --
    name varchar(255),
    startShowDate timestamp,
    periodOfShowing integer,
    description varchar(255),
    --
    primary key (ID)
);