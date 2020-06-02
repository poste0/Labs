package ru.entities;

import javax.persistence.*;
import java.util.List;
import java.util.UUID;

@Table(name = "NODE")
@Entity(name = "Node")
public class Node {
    private static final long serialVersionUID = -6291200724455780790L;

    @Id
    private UUID id;

    @Column(name = "name")
    private String name;

    @Column(name = "address")
    private String address;

    @Column(name = "cpu")
    private String cpu;

    @Column(name = "gpu")
    private String gpu;

    @ManyToMany(mappedBy = "nodes")
    private List<User> users;

    public Node(){
        this.id = UUID.randomUUID();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getCpu() {
        return cpu;
    }

    public void setCpu(String cpu) {
        this.cpu = cpu;
    }

    public String getGpu() {
        return gpu;
    }

    public void setGpu(String gpu) {
        this.gpu = gpu;
    }

    public List<User> getUsers() {
        return users;
    }

    public void setUsers(List<User> users) {
        this.users = users;
    }
}