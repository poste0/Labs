package com.company.enterpriselaba.web.screens.show;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.entity.Theatre;
import com.haulmont.cuba.core.entity.contracts.Id;
import com.haulmont.cuba.core.global.AppBeans;
import com.haulmont.cuba.core.global.DataManager;
import com.haulmont.cuba.core.global.UserSessionSource;
import com.haulmont.cuba.gui.model.DataLoader;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Show;
import com.haulmont.cuba.security.entity.User;

import javax.inject.Inject;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.UUID;
import java.util.stream.Collectors;

@UiController("enterpriselaba_Show.browse")
@UiDescriptor("show-browse.xml")
@LookupComponent("showsTable")
@LoadDataBeforeShow
public class ShowBrowse extends StandardLookup<Show> {
}