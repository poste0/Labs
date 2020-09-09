-- begin ENTERPRISELABA_THEATRE
create table ENTERPRISELABA_THEATRE (
    ID varchar(36) not null,
    VERSION integer not null,
    CREATE_TS timestamp,
    CREATED_BY varchar(50),
    UPDATE_TS timestamp,
    UPDATED_BY varchar(50),
    DELETE_TS timestamp,
    DELETED_BY varchar(50),
    --
    name varchar(255),
    address varchar(255),
    admin_id varchar(36),
    --
    primary key (ID)
)^
-- end ENTERPRISELABA_THEATRE
-- begin ENTERPRISELABA_TICKET
create table ENTERPRISELABA_TICKET (
    ID varchar(36) not null,
    VERSION integer not null,
    CREATE_TS timestamp,
    CREATED_BY varchar(50),
    UPDATE_TS timestamp,
    UPDATED_BY varchar(50),
    DELETE_TS timestamp,
    DELETED_BY varchar(50),
    --
    show_id varchar(36),
    status varchar(255),
    --
    primary key (ID)
)^
-- end ENTERPRISELABA_TICKET
-- begin ENTERPRISELABA_AUDITORIUM
create table ENTERPRISELABA_AUDITORIUM (
    ID varchar(36) not null,
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
    theatre_id varchar(36),
    --
    primary key (ID)
)^
-- end ENTERPRISELABA_AUDITORIUM
-- begin ENTERPRISELABA_SHOW
create table ENTERPRISELABA_SHOW (
    ID varchar(36) not null,
    VERSION integer not null,
    CREATE_TS timestamp,
    CREATED_BY varchar(50),
    UPDATE_TS timestamp,
    UPDATED_BY varchar(50),
    DELETE_TS timestamp,
    DELETED_BY varchar(50),
    --
    name varchar(255),
    film_id varchar(36),
    theatre_id varchar(36),
    auditorium_id varchar(36),
    showDate timestamp,
    price double precision,
    --
    primary key (ID)
)^
-- end ENTERPRISELABA_SHOW
-- begin ENTERPRISELABA_FILM
create table ENTERPRISELABA_FILM (
    ID varchar(36) not null,
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
)^
-- end ENTERPRISELABA_FILM
-- begin SEC_USER
alter table SEC_USER add column DTYPE varchar(100) ^
update SEC_USER set DTYPE = 'enterpriselaba_Admin' where DTYPE is null ^
-- end SEC_USER
