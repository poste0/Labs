package com.company.enterpriselaba.entity;

import com.haulmont.cuba.core.entity.StandardEntity;
import com.haulmont.cuba.core.entity.annotation.Extends;
import com.haulmont.cuba.security.entity.User;

import javax.persistence.Entity;
import javax.persistence.OneToMany;
import javax.persistence.Table;
import java.util.List;

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
}