import { Bot } from "grammy";

const bot = new Bot(process.env.TELEGRAM_BOT_TOKEN);
bot.command("start", (ctx) => ctx.reply("Hello! Bot is connected."));
bot.start();