package com.company.enterpriselaba.entity;

import com.esotericsoftware.kryo.NotNull;
import com.haulmont.cuba.core.entity.StandardEntity;

import javax.persistence.*;
import java.util.List;

@Table(name = "ENTERPRISELABA_AUDITORIUM")
@Entity(name = "enterpriselaba_Auditorium")
public class Auditorium extends StandardEntity {
    private static final long serialVersionUID = 4267641624424359243L;

    @Column(name = "countOfSeats")
    @NotNull
    private Integer countOfSeats;

    @ManyToOne
    @JoinColumn(name = "theatre_id")
    private Theatre theatre;

    @OneToMany(mappedBy = "auditorium")
    private List<Show> shows;

    public Integer getCountOfSeats() {
        return countOfSeats;
    }

    public void setCountOfSeats(Integer countOfSeats) {
        this.countOfSeats = countOfSeats;
    }

    public List<Show> getShows() {
        return shows;
    }

    public void setShows(List<Show> shows) {
        this.shows = shows;
    }

    public Theatre getTheatre() {
        return theatre;
    }

    public void setTheatre(Theatre theatre) {
        this.theatre = theatre;
    }
}