package com.company.enterpriselaba.service;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.entity.Auditorium;
import com.company.enterpriselaba.entity.Theatre;
import com.haulmont.cuba.core.entity.contracts.Id;
import com.haulmont.cuba.core.global.*;
import org.springframework.stereotype.Service;

import javax.inject.Inject;
import java.util.List;
import java.util.UUID;

@Service(TheatreService.NAME)
public class TheatreServiceBean implements TheatreService {
    @Inject
    private DataManager dataManager;

    @Inject
    private Metadata metadata;

    @Override
    public void addTheatre(String name, String address, Admin admin) {
        checkIfAdmin(admin);

        Theatre theatre = metadata.create(Theatre.class);
        theatre.setName(name);
        theatre.setAddress(address);
        theatre.setAdmin(admin);

        dataManager.commit(theatre);
    }

    private void checkIfAdmin(Admin admin) throws IllegalArgumentException{
        if(!AppBeans.get(UserSessionSource.class).getUserSession().getUser().getId().equals(admin.getId())){
            throw new IllegalArgumentException("Admin is not the current admin");
        }
    }

    @Override
    public void changeTheatre(Theatre theatre, String name, String address, Admin admin) {
        checkIfAdmin(admin);

        theatre.setName(name);
        theatre.setAddress(address);
        theatre.setAdmin(admin);

        dataManager.commit(theatre);
    }

    @Override
    public void deleteTheatre(Theatre theatre, Admin admin) {
        checkIfAdmin(admin);

        dataManager.remove(theatre);
    }

    @Override
    public Theatre getTheatre(UUID id) {

        return dataManager.load(Id.of(id, Theatre.class)).one();
    }

    @Override
    public void addAuditorium(Integer countOfSeats, Theatre theatre, Admin admin) {
        checkIfAdmin(admin);

        List<Auditorium> auditoriums = dataManager.loadList(LoadContext.create(Auditorium.class).setQuery(LoadContext.createQuery("SELECT a FROM enterpriselaba_Auditorium a WHERE a.theatre.id = :theatreId").setParameter("theatreId", theatre.getId())));

        Auditorium auditorium = metadata.create(Auditorium.class);
        auditorium.setCountOfSeats(countOfSeats);
        auditorium.setTheatre(theatre);
        auditorium.setNumber(auditoriums.size() + 1);

        dataManager.commit(auditorium);
    }

    @Override
    public void changeAuditorium(Auditorium auditorium, Integer countOfSeats, Admin admin) {
        checkIfAdmin(admin);

        auditorium.setCountOfSeats(countOfSeats);

        dataManager.commit(auditorium);

    }
}