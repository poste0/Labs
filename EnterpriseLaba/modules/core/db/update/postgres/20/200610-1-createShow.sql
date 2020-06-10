create table ENTERPRISELABA_SHOW (
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
    film_id uuid,
    theatre_id uuid,
    auditorium_id uuid,
    showDate timestamp,
    price double precision,
    --
    primary key (ID)
);