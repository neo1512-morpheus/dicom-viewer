import React from 'react';
import { NavBar, Svg, Icon, IconButton } from '../../../components';

const Header = () => {
  return (
    <NavBar className="justify-between border-b-4 border-black">
      <div className="flex flex-1 justify-between">
        <div className="flex items-center">
          <div className="mr-3 inline-flex items-center">
            <Icon
              name="chevron-left"
              className="text-primary-active w-8 cursor-pointer"
              onClick={() => alert('Navigate to previous page')}
            />
            <a
              href="#"
              className="ml-4"
            >
              <Svg name="logo-oroscan" />
            </a>
          </div>
        </div>
        <div className="flex items-center"></div>
        <div className="flex items-center">
          <IconButton
            variant="text"
            color="inherit"
            className="text-primary-active"
            onClick={() => { }}
          >
            <React.Fragment>
              <Icon name="settings" /> <Icon name="chevron-down" />
            </React.Fragment>
          </IconButton>
        </div>
      </div>
    </NavBar>
  );
};

export default Header;
