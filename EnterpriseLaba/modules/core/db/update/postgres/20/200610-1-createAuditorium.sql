create table ENTERPRISELABA_AUDITORIUM (
    ID uuid,
    VERSION integer not null,
    CREATE_TS timestamp,
    CREATED_BY varchar(50),
    UPDATE_TS timestamp,
    UPDATED_BY varchar(50),
    DELETE_TS timestamp,
    DELETED_BY varchar(50),
    --
    countOfSeats integer,
    number integer,
    theatre_id uuid,
    --
    primary key (ID)
);