package com.company.enterpriselaba.web.screens.show;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.entity.Theatre;
import com.company.enterpriselaba.entity.Ticket;
import com.haulmont.cuba.core.entity.Entity;
import com.haulmont.cuba.core.entity.contracts.Id;
import com.haulmont.cuba.core.global.AppBeans;
import com.haulmont.cuba.core.global.DataManager;
import com.haulmont.cuba.core.global.UserSessionSource;
import com.haulmont.cuba.gui.UiComponents;
import com.haulmont.cuba.gui.components.*;
import com.haulmont.cuba.gui.model.DataLoader;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Show;
import com.haulmont.cuba.gui.screen.LookupComponent;
import com.haulmont.cuba.security.entity.User;

import javax.inject.Inject;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.stream.Collectors;

@UiController("enterpriselaba_Show.browse")
@UiDescriptor("show-browse.xml")
@LookupComponent("showsTable")
@LoadDataBeforeShow
public class ShowBrowse extends StandardLookup<Show> {
    @Inject
    private UiComponents uiComponents;

    @Inject
    private Button ticketButton;

    @Inject
    private GroupTable<Show> showsTable;

    public void sellTicketClick() {
    }

    public Component seatsAvailableGenerator(Entity entity) {
        Show show = (Show) entity;

        Label<String> result = uiComponents.create(Label.NAME);
        List<Ticket> tickets = show.getTickets();
        result.setValue(show.getAuditorium().getCountOfSeats() - tickets.size() + "(" + show.getAuditorium().getCountOfSeats() + ")");

        return result;
    }

    @Subscribe
    private void onInit(InitEvent event){
        showsTable.addSelectionListener(new Consumer<Table.SelectionEvent<Show>>() {
            @Override
            public void accept(Table.SelectionEvent<Show> showSelectionEvent) {
                ticketButton.setEnabled(true);
            }
        });

    }
}