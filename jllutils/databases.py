"""
Classes to wrap databases in a more convenient way
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from abc import ABC
from datetime import datetime as dtime
from logging import getLogger
import re
import sqlite3
import time

from numpy import number as npnumber, signedinteger as npint, bool_ as npbool, issubdtype

from .miscutils import all_or_none


logger = getLogger(__name__)


class RowSelectionError(Exception):
    """
    Class to use when unable to correctly select a row
    """
    pass


# The following methods are used to verify and convert Python
# values into SQLite3 values
_db_date_fmt = '%Y-%m-%d'
_db_datetime_fmt = '%Y-%m-%d %H:%M:%S'


def check_date(value, name, date_format):
    """
    Check that a date is or can be formatted into a form SQLite3 accepts

    :param value: the date value
    :type value: :class:`datetime.datetime` or str in format matching ``date_format``

    :param name: the name of the value, to use in error messages
    :type name: str

    :param date_format: the format to require the date in. Must be recognized by :func:`~datetime.datetime.strptime`.
    :type date_format: str

    :return: the date/time formatted into a proper string
    :rtype: str

    :raises ValueError: if given as an improperly formatted string
    :raises TypeError: if given as neither datetime nor string.
    """

    if isinstance(value, dtime):
        # If given as a datetime, just convert to the required string
        return value.strftime(date_format)
    elif isinstance(value, str):
        # If given as a string, we can try converting back to a datetime object to check
        # if the format is correct
        try:
            dtime.strptime(value, date_format)
        except ValueError:
            # If the format is wrong, it raises a ValueError, but we want to use a more
            # recognizable message. We raise "from None" to ignore the traceback of the
            # original error
            raise ValueError('If given as a string, {} must be in {} format'.format(name, date_format)) from None
        else:
            # If formatted correctly, just return the string
            return value
    else:
        raise TypeError('{} must be a datetime instance or a string in {} format'.format(name, date_format))


def check_int(value, name):
    """
    Check that the given value is an integer.

    :param value: value to check

    :param name: name of the value to use in error messages

    :return: the value, if it is an int
    :rtype: int

    :raises TypeError: if the value is not an int
    """
    if not issubdtype(type(value), npint):
        raise TypeError('{} must be an integer'.format(name))
    else:
        return int(value)


def check_real(value, name):
    """
    Check that the given value is a float, or can be made one.

    :param value: value to check

    :param name: name of the value to use in error messages

    :return: the value, if it is a float or can be made one
    :rtype: float

    :raises TypeError: if the value is not a number
    """
    if not issubdtype(type(value), npnumber):
        raise TypeError('{} must be a number'.format(name))
    else:
        return float(value)


def check_bool(value, name):
    """
    Check that the given value is a boolean.

    :param value: value to check

    :param name: name of the value to use in error messages

    :return: the value, if it is a bool
    :rtype: bool

    :raises TypeError: if the value is not a bool
    """
    if not issubdtype(type(value), npbool):
        raise TypeError('{} must be a bool'.format(name))
    else:
        return bool(value)


def check_text(value, name):
    """
    Check that the given value is a string.

    :param value: value to check

    :param name: name of the value to use in error messages

    :return: the value, if it is a str

    :raises TypeError: if the value is not a str
    """
    if not isinstance(value, str):
        raise TypeError('{} must be a string'.format(name))
    else:
        return value


# This dictionary maps SQL types to the check functions.
# Each function must accept two arguments: the value to check
# and the name to use in error messages. Each function must
# return the value that will be inserted in the SQL table.
#
# The idea is that if the methods below have a value to convert
# to an SQL type, they call the function here for that SQL type
# to validate or convert the value as needed.
_sql_var_mapping = {'date': lambda val, name: check_date(val, name, _db_date_fmt),
                    'datetime': lambda val, name: check_date(val, name, _db_datetime_fmt),
                    'integer': check_int,
                    'real': check_real,
                    'boolean': check_bool,
                    'text': check_text}


class SQLSetupError(Exception):
    """
    Class for errors during setup of an SQL table
    """
    pass


class SQLColumnError(Exception):
    """
    Class for errors relating to column names of an SQL table
    """
    pass


class DatabaseTable(ABC):
    """
    This abstract class contains all the functionality common to accessing SQLite3 or MySQL database tables. Concrete
    subclasses for each of those types of databases will have different init methods to set up the connection.
    """

    default_pragmas = {'foreign_keys': 'ON'}

    @property
    def connection(self):
        """
        The connection to the database
        """
        return self._connection

    @property
    def table_name(self):
        """
        The name of the table in the SQL database wrapped by this class
        :rtype: str
        """
        return self._table_name

    @property
    def columns(self):
        """
        The dictionary defining column names and types in the table
        :rtype: dict
        """
        return self._columns

    @property
    def primary_keys(self):
        """
        The tuple of column names that are used as primary keys in the table
        :rtype: tuple
        """
        return self._primary_keys

    def __init__(self, connection, table, columns=None, primary_key_cols=None, modifiers=None,
                 autocommit=False, commit_on_close=True, retries=0, retry_delay=1, delete_existing_table=False,
                 pragmas=None, foreign_keys=None, verbose=0, use_logger=False):
        """
        :param connection: an object that serves as a connection to a database. Must have an ``execute`` method that returns
         a cursor to that database.

        :param table: the name of the table in the database to read
        :type table: str

        :param columns: a dictionary where the keys are the names of columns required in the table and the values are
         strings giving the expected SQL type. Currently supported types are "date", "datetime", "integer", "boolean",
         and "text". May be omitted if connecting to an existing table. If not omitted and connecting to an existing table,
         the column names and types must match the existing table.
        :type columns: dict

        :param primary_key_cols: a tuple or list specifying by name which columns are to be primary keys. If the table is
         being created they will be defined as PRIMARY KEY NOT NULL. May be omitted if connecting to an existing table. If
         not omitted and connecting to an existing table, the list of primary keys must match the existing table.
        :type primary_key_cols: list or tuple of str

        :param modifiers: a dictionary where the keys are the names of columns and the values are strings giving any
         SQL modifiers that that column should have, as a string. E.g. if you want the "file" column to be unique and not
         null, then ``modifiers = {'file': 'UNIQUE NOT NULL'} would accomplish that. Not all columns are required to be
         included in this dictionary; any that are absent are assumed to have no modifiers.
        :type modifiers: dict

        :param autocommit: optional, controls whether every operation on the table should be automatically committed.
         Default is ``False``.
        :type autocommit: bool

        :param commit_on_close: optional, controls whether any pending operations to the table should automatically be
         committed when closing the connection, either through the `close` method or when the class is automatically
         closed by a ``with`` block. Default is ``True``.
        :type commit_on_close: bool

        :param retries: optional, how many times an SQL operation should be retried if there is an OperationalError,
         which usually indicates that the database is locked due to another operation.
        :type retries: int

        :param retry_delay: optional, how long to wait between attempts. Note that this is a minimum delay, it may be
         extended due to slower query operations.
        :type retry_delay: int or float

        :param delete_existing_table: optional, controls whether an existing table with the given name should be deleted.
         Default is ``False``. Use with great caution!
        :type delete_existing_table: bool

        :param pragmas: a dictionary listing SQLite "PRAGMA" statements to execute immediately after the connection is
         established. The keys should be the pragma names, the values the values they should be assigned. For example,
         ``pragmas = {'foreign_keys': 'ON'}`` would execute ``PRAGMA foreign_keys = ON;`` immediately after the connection
         is established. If this argument is not given, default pragmas defined in the class attribute ``default_pragmas``
         will be used. Subclasses can override this attribute to change their default pragmas. To execute no pragmas,
         pass an empty dict.
        :type pragmas: dict

        :param foreign_keys: a dictionary listing what columns are foreign keys. The keys of the dictionary are the columns
         in this table, the values are the foreign table and column in 'table(column)' format. Example:
         ``foreign_keys = {'file_id': 'files(uid)'}`` would be equivalent to SQL: ``FOREIGN KEY(file_id) REFERENCES files(uid)``
         Note that this may require ``{'foreign_keys': 'ON'}`` to be in your ``pragmas``; however, that is present in the
         default pragmas for this class.
        :type foreign_keys:

        :param verbose: controls verbosity of this class. Higher numbers print more debugging statements. Not used if
         ``use_logger`` is ``True``.
        :type verbose: int

        :param use_logger: if ``True``, then any logging that this instance does will be done via the :mod:`logging`
         module. It uses a logger that inherits all properties from the root logger, so by configuring the root logger,
         you can configure what statements are printed and to where.
        :type use_logger: bool
        """
        self._connection = connection
        self._setup_pragmas(pragmas)
        self._table_name = table
        self._autocommit = autocommit
        self._commit_on_close = commit_on_close
        self._retries = retries
        self._retry_delay = retry_delay
        self._verbose = verbose
        self._use_logger = use_logger

        self._columns, self._primary_keys, self._modifiers, self._foreign_keys = self.setup_tables(columns, primary_key_cols, modifiers,
                                                                                                   foreign_keys,
                                                                                                   delete_existing=delete_existing_table)

    def __enter__(self):
        """
        Magic method for ``with...as...`` blocks.
        :return: this instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Magic method for ``with...as...`` blocks
        :return: None
        """
        self.close()

    def close(self):
        """
        Close the connection to the database.

        Commits changes automatically if ``commit_on_close`` was ``True`` during initialization.

        :return: None
        """
        if self._commit_on_close:
            self.commit()

        self._connection.close()

    def log(self, level, msg, *fmtargs, **fmtkwargs):
        """
        Logging method.

        :param level: verbosity level required; self.verbose must be >= to this to print
        :type level: int

        :param msg: The message to print. Will be formatted with the ``.format`` method before printing.
        :type msg: str

        :param fmtargs: Positional arguments for the ``format`` method.

        :param fmtkwargs: Keyword arguments for the ``format`` method.

        :return: None
        """
        msg = msg.format(*fmtargs, **fmtkwargs)
        if self._use_logger:
            if level < 0:
                logger.critical(msg)
            elif level == 0:
                logger.warning(msg)
            elif level == 1:
                logger.info(msg)
            elif level > 1:
                logger.debug(msg)
        elif self._verbose >= level:
            indent = '  ' * level
            print(indent + msg)

    @staticmethod
    def _format_key(k):
        """
        Format a values dict key to work in an SQL values role

        For column names that are SQL reserved keywords (e.g. "or"), they can be enforced to be treated as column
        names by surrounding with brackets ("[or]"). However, when passing a dictionary to self.sql for the "values"
        keyword, the keys cannot have the brackets, so this extracts the key inside the brackets.
        """
        in_bracket = re.search(r'(?<=^\[)\w+(?=\]$)', k)
        if in_bracket:
            return in_bracket.group()

        plain = re.search(r'^\w+$', k)
        if not plain:
            raise SQLColumnError('{} is not a valid column name. Column names must be alphanumerics, optionally '
                                 'enclosed in brackets.'.format(k))
        else:
            return plain.group()

    def add_table_row(self, row_dict):
        """
        Add a new row to the table.

        :param row_dict: a dictionary defining the value each column will take. The column names must be the keys of
            the dictionary, and the corresponding values will go into those columns. This method will try to convert
            types intelligently, e.g. Python :class:`~datetime.datetime` objects will be converted to SQL-compatible
            date strings, and if any value given is incompatible with the SQL type, it will raise an error.
        :type row_dict: dict

        :return: None
        """

        # One problem I ran into was when I was dealing with column names that matched reserved SQL words ("or" in this
        # case). The way to allow that is to put the column names in brackets, however, then the :key syntax can't be
        # used. So we have to take a dictionary with keys that include brackets where necessary, keep the brackets when
        # naming the columns in the tablename() part of the command, but strip them for the :key values.

        row_dict = self._check_row_dict(row_dict)

        # Construct the SQL command that will insert them. According to
        # https://www.pythoncentral.io/introduction-to-sqlite-in-python/,
        # we can use a dictionary of values if the VALUE() part uses key formatting
        columns = list(row_dict.keys())
        keys = ', '.join([':' + self._format_key(c) for c in columns])
        columns = ', '.join(columns)
        command_str = 'INSERT INTO {table}({columns}) VALUES({keys})'
        new_row_dict = {self._format_key(k): v for k, v in row_dict.items()}

        self.sql(command_str, values=new_row_dict, columns=columns, keys=keys)

    def update_row(self, identifying_values, new_values):
        """
        Update a row in the database.

        :param identifying_values: a dictionary with keys specifying the column names and values specifying
         the value that column must have, i.e. this will be expanded into a "WHERE key1 = value1 AND key2 = value2 ..."
         query. The values will be converted to SQL types if necessary/possible.
        :type identifying_values: dict

        :param new_values: a dictionary specifying the new values to insert into the row; the keys give the column
         names. Cannot share any keys with the ``identifying_values`` dict.
        :type new_values: dict

        :return: None
        """
        if any(k in new_values.keys() for k in identifying_values):
            raise ValueError('A column used in identifying values cannot be in new_values as well')

        where_crit_str = self._format_where_crit_string(identifying_values.keys())
        set_str = self._format_set_string(new_values.keys())

        # Like add_table_row, the values dict needs to have any keys in brackets replaced with 
        # unbracketed keys.
        vals = dict()
        vals.update(identifying_values)
        for k, v in new_values.items():
            vals[self._format_key(k)] = v

        self.sql('UPDATE {table} SET {set} WHERE {crit}', values=vals, set=set_str, crit=where_crit_str)

    def fetch_rows_as_dicts(self, identifying_values, columns_to_fetch='*', single_row=False):
        """
        Retrieve certain rows from the database, converting them to dictionaries

        :param identifying_values: a dictionary with keys specifying the column names and values specifying
         the value that column must have, i.e. this will be expanded into a "WHERE key1 = value1 AND key2 = value2 ..."
         query. The values will be converted to SQL types if necessary/possible.
        :type identifying_values: dict

        :param columns_to_fetch: optional, if given, a tuple of column names to include in the dictionary. If
         omitted, all columns are included.
        :type columns_to_fetch: tuple

        :param single_row: optional, controls whether to return a single dictionary (``False``, default)
         or a list of dicts (``True``). If ``True``, then the identifying values must specify exactly one row
         or a :class:`.RowSelectionError` is raised.
        :type single_row: bool

        :return: list of dicts (if ``single_row = False``) or dict (``single_row = True``) representing row(s)
         in the table.
        :rtype: list of dicts or dict
        """
        identifying_values = self._check_row_dict(identifying_values)

        if columns_to_fetch == '*':
            # Can't just keep the columns to fetch as a * - it'll work in the SQL query,
            # but then it'll return the values in the order of the columns in the table,
            # and if that is different than the order of our keys, then the wrong values
            # will go to the wrong keys in the dictionary.
            #
            # Instead, by specifying the order we want the columns in for the SQL query,
            # we guarantee that the values and keys will be matched up correctly.
            columns_to_fetch = self.columns.keys()

        where_crit_str = self._format_where_crit_string(identifying_values.keys())
        if len(identifying_values) > 0:
            cursor = self.sql('SELECT {columns} FROM {table} WHERE {crit};', values=identifying_values,
                              columns=', '.join(columns_to_fetch), crit=where_crit_str)
        else:
            cursor = self.sql('SELECT {columns} FROM {table};', columns=', '.join(columns_to_fetch))

        row_dicts = []
        for row in cursor:
            row_dicts.append(dict(zip(columns_to_fetch, row)))

        if not single_row:
            return row_dicts
        else:
            if len(row_dicts) < 1:
                raise RowSelectionError('No rows found for conditions: {}'.format(identifying_values))
            elif len(row_dicts) > 1:
                raise RowSelectionError('Multiple rows found for conditions: {}'.format(identifying_values))
            else:
                return row_dicts[0]

    def count_rows(self, **conditions):
        """
        Count the number of rows where one or more columns equal the given value(s)

        :param conditions: keyword arguments ``column=value``, which specify what value a given column should have
         for that row to be counted.

        :return: number of rows matching the given conditions
        :rtype: int
        """
        conditions = self._check_row_dict(conditions)
        cmd = 'SELECT COUNT(*) FROM {table} WHERE {conditions}'
        result = self.sql(cmd, conditions=self._format_where_crit_string(conditions.keys()), values=conditions)

        return result.fetchall()[0][0]

    def sql(self, sql_command, values=None, **format_vals):
        """
        Execute an SQL command on the connected database

        :param sql_command: The query to execute. It will be preformatted with:
            * ``{table}`` will be replaced with the table name.
            * additional keyword args are also inserted; e.g. if you include ``{column}``
              in your ``sql_command`` string and add a keyword ``column='NewColumn'``,
              then 'NewColumn' will be inserted where ``{column}`` was.
            Note: it is not recommended to use this to pass values into the table, because
            arbitrary commands could be sent. For data values, use the ``values`` input.
            However, some parts of the command cannot be inserted with the `sqlite3`
            values method.
        :type sql_command: str

        :param values: values to insert in the SQL query, using either the ``?`` or ``:key``
            notation with a in the command string with a sequence or dict, respectively
            (see `the docs <https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.execute>`_
            for more information)
        :type values: tuple or dict

        :param format_vals: additional keyword arguments will be passed through to the
            format statement that does the initial formatting of the SQL command (i.e.
            that adds the table name).

        :return: the cursor that results from executing the command
        :rtype: :class:`sqlite3.Cursor`
        """
        if values is not None and not isinstance(values, (tuple, dict)):
            raise TypeError('values must be a tuple or dict')

        sql_command = sql_command.format(table=self.table_name, **format_vals)
        sql_errors_to_catch = (sqlite3.OperationalError, sqlite3.IntegrityError)
        if values is None:
            exargs = []
            err_fmt = '\nSQL command was "{cmd}"'

        else:
            exargs = [values]
            err_fmt = '\nSQL command was "{cmd}" with values = {vals}'

        ntries = 0

        while True:
            try:
                cursor = self.connection.cursor().execute(sql_command, *exargs)
            except sql_errors_to_catch as err:
                if isinstance(err, sqlite3.OperationalError) and 'locked' in err.args[0] and ntries < self._retries:
                    ntries += 1
                    self.log(1, 'Operational error: waiting {} sec ({} retries remaining)',
                             self._retry_delay, self._retries - ntries)
                    time.sleep(self._retry_delay)
                    continue
                msg = err.args[0] + err_fmt.format(cmd=sql_command, vals=values)
                raise err.__class__(msg) from None
            else:
                break

        if self._autocommit:
            # Not sure if this will cause a problem with commands that don't actually change anything
            self.commit()

        return cursor

    def commit(self):
        """
        Commit and pending changes to the database.
        :return: None
        """
        self._connection.commit()

    #################
    # Setup methods #
    #################
    def setup_tables(self, columns, primary_keys, modifiers, foreign_keys=None, delete_existing=False):
        """
        A method that initializes any required tables.

        By default, this ensures that the table exists and has the columns and primary keys specified. If no columns
        or keys are specified, they are read from the table.

        :param columns: the dictionary defining column names and types, or None
        :type columns: dict or None

        :param primary_keys: the tuple defining which columns are primary keys, or None
        :type primary_keys: tuple, list, or None

        :param modifiers: the dictionary defining extra modifiers for columns, or None.
        :type modifiers: dict or None

        :param delete_existing: controls whether to delete an existing table and start from scratch. Default is
         ``False``.
        :type delete_existing: bool

        :return: column dictionary, primary keys tuple, and modifiers dictionary
        :rtype: dict, tuple, dict
        """
        def columns_are_same(col_in, col_table):
            return all([c in col_in for c in col_table]) and all([c in col_table for c in col_in])

        def types_are_same(col_in, col_table):
            # do case-insensitive check because the column types may be upper or lower case in the SQL database
            # assuming that both have the same keys; columns_are_same should be called first
            return all([col_in[k].lower() == col_table[k].lower() for k in col_in.keys()])

        def fks_are_same(fks_in, fks_table):
            # If there are no given foreign keys, then allow the list of ones from the table to be either None
            # itself or an empty dict.
            if fks_in is None:
                return fks_table is None or len(fks_table) == 0

            # Ensure the same foreign keys are present both in the input and in the existing table.
            # Make the comparison case-insensitve just in case.
            fks_in = {k.lower(): v.lower() for k, v in fks_in.items()}
            fks_table = {k.lower(): v.lower() for k, v in fks_table.items()}
            return fks_in == fks_table

        if delete_existing:
            self.sql('DROP TABLE IF EXISTS {table};')
            self.commit()

        table_exists = self._check_columns(columns, primary_keys, modifiers)

        # If there's no existing table, _check_columns has made sure that the columns are given
        if not table_exists:
            if foreign_keys is None:
                command_str = 'CREATE TABLE IF NOT EXISTS {table} ({columns});'
            else:
                command_str = 'CREATE TABLE IF NOT EXISTS {table} ({columns}, {fkeys});'
            self.sql(command_str, columns=self._format_column_names_types(columns, primary_keys, modifiers),
                     fkeys=self._format_foreign_keys(foreign_keys))

        # Check that all the columns are present
        table_columns, table_keys, table_modifiers, table_foreign_keys = self._get_table_columns()
        if columns is None:
            columns = table_columns
        else:
            # check that all table columns are in the given columns and have the same type.
            # eventually, can add the ability to add new columns
            if not columns_are_same(columns, table_columns):
                raise SQLSetupError('Given columns are different than the columns already in the table')
            elif not types_are_same(columns, table_columns):
                raise SQLSetupError('Given columns have different types than the columns already in the table')

        if primary_keys is None:
            primary_keys = table_keys
        else:
            # verify that the given primary keys and extant primary keys are the same
            if not columns_are_same(primary_keys, table_keys):
                raise SQLSetupError('Given primary keys are different than the keys in use in the table')

        if modifiers is None:
            modifiers = table_modifiers
        else:
            if not columns_are_same(modifiers, table_modifiers) or not types_are_same(modifiers, table_modifiers):
                raise SQLSetupError('Given modifiers differ from those in the existing table')

        if not fks_are_same(foreign_keys, table_foreign_keys):
            raise SQLSetupError('Given foreign keys differ from those in the existing table')

        return columns, primary_keys, modifiers, foreign_keys

    def _setup_pragmas(self, pragmas):
        if pragmas is None:
            pragmas = self.default_pragmas

        for k, v in pragmas.items():
            self.connection.execute('PRAGMA {} = {};'.format(k, v))

    ############################
    # Table formatting methods #
    ############################
    def _check_columns(self, columns, primary_keys, modifiers):
        """
        Verify that the given columns/primary keys are compatible with existing ones

        :param columns: the dictionary of column names and types requested, or None
        :type columns: dict or None

        :param primary_keys: the list or tuple of primary keys, or None
        :type primary_keys: list, tuple, or None

        :param modifiers: the dictionary defining extra modifiers for columns, or None.
        :type modifiers: dict or None

        :return: boolean indicating if the table already exists in the database
        :rtype: bool
        """
        # If we're creating a new table, either because it never existed before or we dropped it, we must have columns
        # and primary keys. Otherwise, they can be None and we will read them from the existing table.
        cur = self.sql("SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")

        no_table = len(cur.fetchall()) == 0
        if no_table:
            # no table
            if columns is None or primary_keys is None:
                raise SQLSetupError(
                    'Creating a new table requires that the columns and primary keys to include be given')

        # Check that all the types in the columns dict have a defined conversion/check function
        if columns is not None:
            for k, v in columns.items():
                if v not in _sql_var_mapping.keys():
                    raise SQLSetupError('Type "{type}" for column "{col}" is not permitted. Allowed types are '
                                        '{allowed}'.format(type=v, col=k, allowed=', '.join(_sql_var_mapping.keys())))

        # If primary_keys is given, columns must be as well
        if primary_keys is not None:
            if columns is None:
                raise SQLSetupError('If primary_keys is given, columns must be given as well')
            else:
                # Check that all the primary keys are defined as columns
                missing_keys = []
                for k in primary_keys:
                    if k not in columns:
                        missing_keys.append(k)
                if len(missing_keys) > 0:
                    raise SQLSetupError('The following primary key(s) are not defined in the column dictionary: {}'.format(
                        ', '.join(missing_keys)
                    ))

        # modifiers is always optional, but if it is given, columns must be given and all keys of modifiers must
        # also be in columns. modifiers are ignored if not creating a new table, so only check if we are creating
        # a table.
        if modifiers is not None:
            if columns is None:
                raise SQLSetupError('If modifiers is given, columns must be as well')
            else:
                key_check = [k for k in modifiers.keys() if k not in columns]
                if len(key_check) > 0:
                    raise SQLSetupError('The following columns used in modifiers are not defined in columns: {}'.format(
                        ', '.join(key_check)
                    ))

        return not no_table

    def _get_table_columns(self, fkfmt='str'):
        """
        Get the existing columns, primary keys, and modifiers in the table.

        :return: column name-type dictionary, tuple of primary keys, modifiers dictionary
        :rtype: dict, tuple, dict
        """
        def fk_name(fkdef):
            return re.search(r'(?<=KEY\()\w+(?=\))', fkdef, re.IGNORECASE).group()

        def fk_ref(fkdef):
            match = re.search(r'(?<=REFERENCES)\s+(?P<table>\w+)\((?P<column>\w+)\)', fkdef)
            if fkfmt == 'str':
                return match.group().strip()
            elif fkfmt == 'dict':
                return {'table': match.group('table'), 'column': match.group('column')}

        cur = self.sql('SELECT sql FROM sqlite_master WHERE name="{table}"')
        create_cmd = cur.fetchall()[0][0]
        # It seems that the "sql" column of the master table is always the CREATE command that would make that table.
        # Even if the table has been altered to add a column after the initial command, this is the case. An example is
        #
        #   CREATE TABLE secondtable(keycol INTEGER PRIMARY KEY NOT NULL, datcol INTEGER, thing text)
        #
        # So to find the column names we need to find the part inside the parentheses, split it up by commas, then get
        # the first word in each section as the column name and the second as the type.
        match = re.search(r'(?<=\().+(?=\))', create_cmd).group()
        
        # credit https://stackoverflow.com/a/26634150 - need to avoid splitting on commas inside parentheses
        # for constraints like CHECK(column in ("alpha", "bravo"))
        column_defs = re.split(r',\s*(?![^()]*\))', match)
        columns = {c.split()[0]: c.split()[1] for c in column_defs if c.split()[0].upper() != 'FOREIGN'}
        primary_keys = [c.split()[0] for c in column_defs if 'PRIMARY KEY' in c]

        # If there's modifiers other than "PRIMARY KEY NOT NULL", we want to get those as well.
        no_prim_keys = match.replace('PRIMARY KEY NOT NULL', '')
        column_defs = re.split(r',\s*(?![^()]*\))', no_prim_keys)
        mod_lists = {c.split()[0]: c.split()[2:] for c in column_defs if c.split()[0].upper() != 'FOREIGN'}
        modifiers = {k: ' '.join(v) for k, v in mod_lists.items() if len(v) > 0}

        # Also list foreign keys
        foreign_key_defs = [' '.join(c.split()[1:]) for c in column_defs if c.split()[0].upper() == 'FOREIGN']
        foreign_keys = {fk_name(fk): fk_ref(fk) for fk in foreign_key_defs}

        return columns, tuple(primary_keys), modifiers, foreign_keys

    def _format_column_names_types(self, columns=None, primary_keys=None, modifiers=None):
        """
        Create the string that defines the columns, their types, and any modifiers, to use in table initialization.

        :param columns: the column name-type dictionary. If omitted, taken from the instance, if possible.
        :type columns: dict

        :param primary_keys: the list or tuple of primary key column names. If omitted, taken from the instance if
         possible.
        :type primary_keys: list or tuple

        :param modifiers: the dictionary defining extra modifiers for columns. If omitted, taken from the instance if
         possible.
        :type modifiers: dict or None

        :return: single string with entries "NAME TYPE [MODIFIERS]" separated by commas. [MODIFIERS] will be "PRIMARY
         KEY NOT NULL" if the column is a primary key, and nothing otherwise.
        :rtype: str
        """

        # Allow this function to be called and pass columns and primary_keys, or to omit those from the function call
        # and get them from the class instance. Require that both are either given or taken from the instance.
        none_inputs = [x is None for x in (columns, primary_keys)]
        if not all_or_none(none_inputs):
            raise TypeError('Give all or none of columns, primary_keys, and modifiers')
        elif all(none_inputs):
            # If both are omitted, get them from the instance.
            try:
                columns = self._columns
                primary_keys = self._primary_keys
                modifiers = self._modifiers
            except AttributeError:
                raise RuntimeError('Tried to call _format_column_names_types without arguments before the instance '
                                   '_columns and _primary_keys attributes are set')

        column_names = []
        for k, v in columns.items():
            col_def = '{} {}'.format(k, v.upper())
            if k in primary_keys:
                col_def += ' PRIMARY KEY NOT NULL'
            if k in modifiers:
                this_mod = modifiers[k]
                # Uppercase most of the modifier, but don't uppercase string literals
                # e.g. CHECK(column in ("alpha", "beta")) should not uppercase the 
                # "alpha" and "beta"
                upper_mod = this_mod.upper()
                strings = re.findall(r'"[^"]+"', this_mod)
                for substr in strings:
                    upper_mod = upper_mod.replace(substr.upper(), substr)
                col_def += ' ' + upper_mod
            column_names.append(col_def)

        return ', '.join(column_names)

    @staticmethod
    def _format_foreign_keys(foreign_keys):
        if foreign_keys is None:
            return None

        fks = []
        for k, v in foreign_keys.items():
            fks.append('FOREIGN KEY({}) REFERENCES {}'.format(k, v))

        return ', '.join(fks)

    def _check_row_dict(self, row_dict):
        """
        Verify that the entries in a dictionary representing one row are the correct types, converting if possible

        :param row_dict: a dictionary with column names as keys and column values as values.
        :type row_dict: dict

        :return: a copy of the row dictionary, with values converted if necessary and possible
        :rtype: dict
        """
        row_dict = row_dict.copy()
        # Go through all of the values, checking that they are the right type, or converting as allowed.
        for key, val in row_dict.items():
            if val is None:
                continue
            req_type = self.columns[key]
            row_dict[key] = _sql_var_mapping[req_type](val, key)

        return row_dict

    @staticmethod
    def _format_where_crit_string(keys):
        """
        Format an SQL WHERE criteria string for a set of dictionary keys

        This can be used in an SQL WHERE clause in conjuction with the SQL module `execute`'s ability to accept a
        dictionary of values to insert in the SQL command. The idea is that if you want to find rows where, say,
        col1 = 1 and col2 = 0, you could do::

            >>> crit = {'col1': 1, 'col2': 0}
            >>> critstr = _format_where_crit_string(crit.keys())
            >>> cursor.execute('SELECT * FROM table WHERE {}'.format(critstr), crit)

        ``critstr`` would be ``col1 = :col1, col2 = :col2``, so `execute` would take the values for those
        columns from the ``crit`` dictionary and insert their sanitized versions into the command string.

        :param keys: a collection of column names to set up as the criteria
        :type keys: collection of str

        :return: a string with the format "key1 = :key1 AND key2 = :key2 AND ...".
        """
        return ' AND '.join(['{k} = :{k}'.format(k=k) for k in keys])

    @classmethod
    def _format_set_string(cls, keys):
        return ', '.join('{} = :{}'.format(k, cls._format_key(k)) for k in keys)


class SQLiteDatabaseTable(DatabaseTable):
    """
    A subclass of DatabaseTable that automatically opens a connection to a SQLite3 database file

    :param database_file: the base to the .sqlite3 database file to open
    :type database_file: str

    :param table: the name of the table in the .sqlite3 file to access
    :type table: str

    For all other parameters see the documentation for `DatabaseTable`.
    """
    def __init__(self, database_file, table, *args, **kwargs):
        conn = sqlite3.connect(database_file)
        super(SQLiteDatabaseTable, self).__init__(conn, table, *args, **kwargs)
