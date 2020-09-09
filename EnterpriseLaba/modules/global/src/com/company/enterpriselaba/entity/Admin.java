package com.company.enterpriselaba.entity;

import com.haulmont.cuba.core.entity.StandardEntity;
import com.haulmont.cuba.core.entity.annotation.Extends;
import com.haulmont.cuba.security.entity.User;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.OneToMany;
import javax.persistence.Table;
import javax.validation.constraints.NotNull;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Objects;

@Extends(User.class)
@Entity(name = "enterpriselaba_Admin")
public class Admin extends User {
    private static final long serialVersionUID = 8110360449125053472L;

    @OneToMany(mappedBy = "admin")
    private List<Theatre> theatres;

    public List<Theatre> getTheatres() {
        return theatres;
    }

    public void setTheatres(List<Theatre> theatres) {
        this.theatres = theatres;
    }

    public void addTheatre(Theatre theatre){
        if(!Objects.isNull(theatres)) {
            theatres.forEach(theatre1 -> {
                if (theatre.getName().equals(theatre1.getName()) && theatre.getAddress().equals(theatre1.getAddress())) {
                    throw new IllegalArgumentException("The theatre already exists");
                }

                theatres.add(theatre);
            });
        }
        else{
            theatres = Arrays.asList(theatre);
        }
    }

}