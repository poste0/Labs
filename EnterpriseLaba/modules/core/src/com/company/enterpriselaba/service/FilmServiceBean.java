package com.company.enterpriselaba.service;

import com.company.enterpriselaba.entity.*;
import com.haulmont.cuba.core.global.*;
import org.springframework.stereotype.Service;

import javax.inject.Inject;
import java.sql.Time;
import java.util.Date;
import java.util.List;

@Service(FilmService.NAME)
public class FilmServiceBean implements FilmService {
    @Inject
    private DataManager dataManager;

    @Inject
    private Metadata metadata;

    @Override
    public String getFilmInfo(Film film) {
        StringBuilder filmInfoBuilder = new StringBuilder();
        filmInfoBuilder.append("Film name: ").append(film.getName()).append("\n");
        filmInfoBuilder.append("Film has been shown since: ").append(film.getStartShowDate().toString()).append("\n");
        filmInfoBuilder.append("The film is shown : ").append(film.getPeriodOfShowing()).append(" days\n");
        filmInfoBuilder.append("The film's description: ").append(film.getDescription()).append("\n");
        filmInfoBuilder.append("The film's shows: ");
        film.getShows().forEach(show -> filmInfoBuilder.append(show.getName()));
        filmInfoBuilder.append("\n");

        return filmInfoBuilder.toString();
    }

    @Override
    public void addFilm(String name, Date startShowDate, Integer periodOfShowing, String description, Admin admin) {
        List<Film> films = dataManager.loadList(LoadContext.create(Film.class).setQuery(LoadContext.createQuery("SELECT f FROM enterpriselaba_Film f")));
        films.forEach(film -> {
            if (film.getName().equals(name)) {
                throw new IllegalArgumentException("The film with the same name already exists");
            }
        });

        checkFilmParams(startShowDate, periodOfShowing, admin);

        Film film = metadata.create(Film.class);
        film.setName(name);
        film.setDescription(description);
        film.setPeriodOfShowing(periodOfShowing);
        film.setStartShowDate(startShowDate);

        dataManager.commit(film);
    }

    private void checkFilmParams(Date startShowDate, Integer periodOfShowing, Admin admin){
        if(startShowDate.compareTo(new Date()) < 0){
            throw new IllegalArgumentException("Date of start is before now");
        }

        if(periodOfShowing <= 0){
            throw new IllegalArgumentException("Period of showing is not more 0");
        }

        if(!AppBeans.get(UserSessionSource.class).getUserSession().getUser().getId().equals(admin.getId())){
            throw new IllegalArgumentException("Admin is not the current admin");
        }
    }

    @Override
    public void changeFilm(Film film, String name, Date startShowDate, Integer periodOfShowing, String description, Admin admin) {
        checkFilmParams(startShowDate, periodOfShowing, admin);

        film.setName(name);
        film.setStartShowDate(startShowDate);
        film.setPeriodOfShowing(periodOfShowing);
        film.setDescription(description);

        dataManager.commit(film);
    }

    @Override
    public void deleteFilm(Film film) {
        dataManager.remove(film);
    }

    @Override
    public void addShow(String name, Film film, Theatre theatre, Auditorium auditorium, Date showDate, Double price, Admin admin) {
        Show show = metadata.create(Show.class);

        checkShowParams(showDate, price, admin);

        show.setName(name);
        show.setFilm(film);
        show.setTheatre(theatre);
        show.setAuditorium(auditorium);
        show.setShowDate(showDate);
        show.setPrice(price);

        dataManager.commit(show);
    }

    private void checkShowParams(Date showdate, Double price, Admin admin){
        if(showdate.before(new Date())) {
            throw new IllegalArgumentException("Date is before now");
        }

        if(price < 0){
            throw new IllegalArgumentException("Price is negative");
        }

        if(!AppBeans.get(UserSessionSource.class).getUserSession().getUser().getId().equals(admin.getId())){
            throw new IllegalArgumentException("Admin is not current");
        }
    }

    @Override
    public void deleteShow(Show show, Admin admin) {
     if(!AppBeans.get(UserSessionSource.class).getUserSession().getUser().getId().equals(admin.getId())){
         throw new IllegalArgumentException("Admin is not current");
     }

     dataManager.remove(show);
    }

    @Override
    public void changeShow(Show show, String name, Film film, Theatre theatre, Auditorium auditorium, Date showDate, Double price, Admin admin) {
        checkShowParams(showDate, price, admin);

        show.setName(name);
        show.setFilm(film);
        show.setTheatre(theatre);
        show.setAuditorium(auditorium);
        show.setShowDate(showDate);
        show.setPrice(price);

        dataManager.commit(show);
    }
}