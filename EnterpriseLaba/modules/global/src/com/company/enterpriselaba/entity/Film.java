package com.company.enterpriselaba.entity;

import com.esotericsoftware.kryo.NotNull;
import com.haulmont.cuba.core.entity.StandardEntity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.OneToMany;
import javax.persistence.Table;
import java.util.Date;
import java.util.List;

@Table(name = "ENTERPRISELABA_FILM")
@Entity(name = "enterpriselaba_Film")
public class Film extends StandardEntity {
    private static final long serialVersionUID = -2641558905078766037L;

    @Column(name = "name")
    @NotNull
    private String name;

    @Column(name = "startShowDate")
    @NotNull
    private Date startShowDate;

    @Column(name = "periodOfShowing")
    @NotNull
    private Integer periodOfShowing;

    @Column(name = "description")
    @NotNull
    private String description;

    @OneToMany(mappedBy = "film")
    private List<Show> shows;

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Integer getPeriodOfShowing() {
        return periodOfShowing;
    }

    public void setPeriodOfShowing(Integer periodOfShowing) {
        this.periodOfShowing = periodOfShowing;
    }

    public Date getStartShowDate() {
        return startShowDate;
    }

    public void setStartShowDate(Date startShowDate) {
        this.startShowDate = startShowDate;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<Show> getShows() {
        return shows;
    }

    public void setShows(List<Show> shows) {
        this.shows = shows;
    }
}