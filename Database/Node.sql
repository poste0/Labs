create table node (
	ID varchar(36) not null,
	name varchar(255),
    	address varchar(255),
    	cpu varchar(255),
    	gpu varchar(255),
    
    	primary key (ID)
);
create table user_node (
	user_id varchar(36) not null,
    	node_id varchar(36) not null,
    	primary key (user_id, node_id)
);	
alter table user_node add constraint FK_user_node_ON_USER foreign key (user_id) references LABA_USER(ID)
alter table user_node add constraint FK_user_node_ON_NODE foreign key (node_id) references NODE(ID)

