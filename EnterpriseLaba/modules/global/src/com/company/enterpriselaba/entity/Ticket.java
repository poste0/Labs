package com.company.enterpriselaba.entity;

import com.esotericsoftware.kryo.NotNull;
import com.haulmont.cuba.core.entity.StandardEntity;

import javax.persistence.*;

@Table(name = "ENTERPRISELABA_TICKET")
@Entity(name = "enterpriselaba_Ticket")
public class Ticket extends StandardEntity {
    private static final long serialVersionUID = 7568656526607032986L;

    @ManyToOne
    @JoinColumn(name = "show_id")
    private Show show;

    @Column(name = "status")
    @NotNull
    private String status;

    public Show getShow() {
        return show;
    }

    public void setShow(Show show) {
        this.show = show;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }
}